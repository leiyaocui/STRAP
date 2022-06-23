import os
from PIL import Image
import numpy as np
import torch


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad()
def fast_hist(pred, target, n):
    k = (target >= 0) & (target < n)
    return torch.bincount(n * target[k].int() + pred[k], minlength=n ** 2).reshape(n, n)


@torch.no_grad()
def per_class_iu(hist):
    return torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist))


@torch.no_grad()
def mIoU(output, target):
    num_classes = output.shape[1]
    _, pred = output.max(dim=1)
    hist = torch.zeros((num_classes, num_classes), device=output.device)
    hist += fast_hist(pred.flatten(), target.flatten(), num_classes)
    ious = per_class_iu(hist) * 100
    return ious.cpu()


class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    @torch.no_grad()
    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    @torch.no_grad()
    def _min_norm_2d(vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, j)] += torch.mul(vecs[i][k], vecs[j][k]).sum()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, i)] += torch.mul(vecs[i][k], vecs[i][k]).sum()
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.mul(vecs[j][k], vecs[j][k]).sum()
                c, d = MinNormSolver._min_norm_element_from2(
                    dps[(i, i)], dps[(i, j)], dps[(j, j)]
                )
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    @torch.no_grad()
    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y.cpu()), axis=0)
        tmpsum = 0.0
        tmax_f = (torch.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return torch.maximum(y - tmax_f, torch.zeros(y.shape, device=y.device))

    @torch.no_grad()
    def _next_point(cur_val, grad, n):
        proj_grad = grad - (torch.sum(grad) / n)

        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = 1.0 - cur_val[proj_grad > 0] / proj_grad[proj_grad > 0]

        # if type(tm1) is torch.Tensor:
        # tm1 = tm1.numpy()
        # tm2 = tm2.numpy()
        # skippers = torch.sum(tm1 < 1e-7) + torch.sum(tm2 < 1e-7)
        t = 1
        if len(tm1[tm1 > 1e-7]) > 0:
            t = torch.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, torch.min(tm2[tm2 > 1e-7]))

        next_point = proj_grad * t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    @torch.no_grad()
    def find_min_norm_element(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        device = vecs[0][0].device

        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)
        n = len(vecs)
        sol_vec = torch.zeros(n, device=device)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = torch.zeros((n, n), device=device)
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0 * torch.matmul(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            change = new_sol_vec - sol_vec
            if torch.sum(torch.abs(change)).item() < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec


@torch.no_grad()
def save_image(data, file_name, save_dir):
    save_path = os.path.join(save_dir, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = data.cpu().numpy().squeeze()
    img = Image.fromarray(data.astype(np.uint8))
    img.save(save_path)


@torch.no_grad()
def save_colorful_image(data, file_name, save_dir, palettes):
    save_path = os.path.join(save_dir, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = data.cpu().numpy().squeeze()
    img = Image.fromarray(palettes[data])
    img.save(save_path)


CITYSCAPE_PALETTE = np.asarray(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [0, 0, 0],
    ],
    dtype=np.uint8,
)


NYU40_PALETTE = np.asarray(
    [
        [0, 0, 0],
        [0, 0, 80],
        [0, 0, 160],
        [0, 0, 240],
        [0, 80, 0],
        [0, 80, 80],
        [0, 80, 160],
        [0, 80, 240],
        [0, 160, 0],
        [0, 160, 80],
        [0, 160, 160],
        [0, 160, 240],
        [0, 240, 0],
        [0, 240, 80],
        [0, 240, 160],
        [0, 240, 240],
        [80, 0, 0],
        [80, 0, 80],
        [80, 0, 160],
        [80, 0, 240],
        [80, 80, 0],
        [80, 80, 80],
        [80, 80, 160],
        [80, 80, 240],
        [80, 160, 0],
        [80, 160, 80],
        [80, 160, 160],
        [80, 160, 240],
        [80, 240, 0],
        [80, 240, 80],
        [80, 240, 160],
        [80, 240, 240],
        [160, 0, 0],
        [160, 0, 80],
        [160, 0, 160],
        [160, 0, 240],
        [160, 80, 0],
        [160, 80, 80],
        [160, 80, 160],
        [160, 80, 240],
    ],
    dtype=np.uint8,
)


AFFORDANCE_PALETTE = np.asarray([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
