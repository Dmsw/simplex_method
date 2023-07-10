import numpy as np


class LinearProblem:
    def __init__(self, A, b, c):
        """
        used to solve the linear constrain optimize problem.
        min     c*x
        s.t.    Ax=b
        :param A: shape (m, n)
        :param b: shape (m, 1)
        :param c: shape (n, 1)
        """
        self.x = np.zeros_like(c, dtype=float)
        self.A = np.array(A)
        self.x_dim = self.A.shape[1]
        self.b = np.array(b).reshape([-1, 1])
        self.b_dim = self.b.shape[0]
        self.c = np.array(c).reshape([-1, 1])
        self.no_solution = False
        self.minimum = False
        self.find_solution = False
        m = np.concatenate([self.A, np.eye(self.b_dim), self.b], axis=1)
        init_zeta = np.concatenate([np.zeros([1, self.x_dim]), -np.ones([1, self.b_dim]), np.zeros([1, 1])], axis=1)
        self.m = np.concatenate([init_zeta, m], axis=0)
        self.indexes = list(range(self.x_dim+self.b_dim))
        self.base_index = list(range(self.x_dim, self.x_dim+self.b_dim))
        self.artificial_index = set(self.base_index)
        self.log = [[self.m, self.base_index]]

    def solve(self):
        print("Finding the initial base vector...")
        self.clear_zeta()
        while not self.minimum:
            self.step()
            if self.no_solution:
                return None

        if self.get_solution_val() > 0:
            self.no_solution = True
            print("No solution found")
            return None

        self.get_init_base_index()
        self.m[0, :-1] = -self.c.T
        self.m[0, -1] = 0
        self.clear_zeta()
        while not self.minimum:
            self.step()
            if self.no_solution:
                print("No minimum solution existed")
                return None

        self.find_solution = True
        self.get_solution()

    def get_solution(self):
        self.x -= self.x
        for i, bi in enumerate(self.base_index):
            self.x[bi] = self.m[i+1, -1]
        return self.x

    def get_solution_val(self):
        return self.m[0, -1]

    def get_init_base_index(self):
        row_tobe_delete = []
        for i, bi in enumerate(self.base_index):
            if bi in self.artificial_index:
                flag = 0
                for j, zj in enumerate(self.m[i+1, :self.x.shape[0]]):
                    if zj > 0:
                        self.base_index[i] = self.indexes[j]
                        self.scale_col_by_row(j, i)
                        flag = 1
                        break
                if not flag:
                    row_tobe_delete.append(i+1)
        if len(row_tobe_delete) > 0:
            self.m = np.delete(self.m, row_tobe_delete, axis=0)
            tobe_delete = np.array(self.base_index)[np.array(row_tobe_delete).astype(int)-1]
            for i in tobe_delete:
                self.base_index.remove(i)

        self.m = np.delete(self.m, list(self.artificial_index), axis=1)
        self.indexes = self.indexes[:-len(self.artificial_index)]

    def scale_col_by_row(self, c, r):
        self.m[r, :] /= self.m[r, c]
        for i in range(self.m.shape[0]):
            if i == r:
                pass
            else:
                self.m[i, :] -= self.m[r, :]*self.m[i, c]

    def step(self):
        index_to_be_push = -1
        index_to_be_pop = -1
        min_theta = np.inf
        for i, zi in enumerate(self.m[0, :-1]):
            if zi > 0 and i not in self.artificial_index:
                index_to_be_push = i
                break

        if index_to_be_push < 0:
            self.minimum = True
            return

        for i, ai in enumerate(self.m[1:, index_to_be_push]):
            if ai > 0:
                theta = self.m[i+1, -1]/ai
                if theta < min_theta:
                    min_theta = theta
                    index_to_be_pop = i

        if index_to_be_pop < 0:
            self.no_solution = True

        self.scale_col_by_row(index_to_be_push, index_to_be_pop+1)
        self.log.append([self.m, self.base_index])
        self.base_index[index_to_be_pop] = self.indexes[index_to_be_push]

    def clear_zeta(self):
        for i, bi in enumerate(self.base_index):
            self.m[i+1, :] /= self.m[i+1, bi]
            self.m[0, :] -= self.m[i+1, :]*self.m[0, bi]

    def print(self):
        print(self.A)
        print(self.b)
        print(self.c)
        print(self.m)

    def print_base_index(self):
        print(self.base_index)

    def print_solution(self):
        print(self.get_solution())

    def print_log(self):
        print(self.log)


if __name__ == '__main__':
    lp = LinearProblem(np.array([[1, 1, 1],
                                 [-1, 1, 0]]),
                       np.array([[1],
                                 [0.5]]),
                       np.array([[-1],
                                 [-1],
                                 [-1]]))
    lp.solve()
    lp.print()
    lp.print_base_index()
    print(lp.get_solution())
