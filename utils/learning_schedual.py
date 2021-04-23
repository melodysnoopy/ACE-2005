import numpy as np


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1 - decay_rate) ** epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return optimizer


class LearningSchedual(object):
    def __init__(self, optimizer, epochs, train_steps, lr, stop_epoch=None):
        self.optimizer = optimizer
        self.train_steps = train_steps
        self.epochs = epochs  # !!!!!!!!!!!!!!!!!!!!!!!
        self.lr = lr
        self.stop_epoch = stop_epoch

        self.warm_steps = 1
        self.all_steps_without_warm = epochs * train_steps - self.warm_steps

        string = "init "
        for i, key in enumerate(lr.keys()):
            self.optimizer.param_groups[i]['lr'] = self.lr[key] * 1 / self.warm_steps
            string += '{} lr: {}, '.format(key, self.optimizer.param_groups[i]['lr'])

        print(string)

    def update_lr(self, epoch, step):
        if self.stop_epoch and epoch >= self.stop_epoch:
            pass
        else:
            global_step = epoch * self.train_steps + step
            global_step_without_warm_step = epoch * self.train_steps + step - self.warm_steps
            if global_step < self.warm_steps:
                for i, key in enumerate(self.lr.keys()):
                    self.optimizer.param_groups[i]['lr'] = self.lr[key] * global_step / self.warm_steps
            elif global_step == self.warm_steps:
                for i, key in enumerate(self.lr.keys()):
                    self.optimizer.param_groups[i]['lr'] = self.lr[key]
            elif step == 1:
                rate = (1 - global_step_without_warm_step / self.all_steps_without_warm)
                for i, key in enumerate(self.lr.keys()):
                    self.optimizer.param_groups[i]['lr'] = self.lr[key] * rate
        lr = {}
        for i, key in enumerate(self.lr.keys()):
            lr[key] = self.optimizer.param_groups[i]['lr']

        return lr


class CosineAnnealingWithWarmRestart(object):
    def __init__(self, optimizer, epochs, train_steps, lr):
        self.optimizer = optimizer
        self.train_steps = train_steps
        self.epochs = epochs
        self.lr = lr

        self.warm_steps = 2400  # 2400
        self.all_steps_without_warm = epochs * train_steps - self.warm_steps
        self.optimizer.param_groups[0]['lr'] = self.lr[0] * 1 / self.warm_steps
        self.optimizer.param_groups[1]['lr'] = self.lr[1] * 1 / self.warm_steps

        print("init small lr:{} large lr:{}".format(self.optimizer.param_groups[0]['lr'],
                                                    self.optimizer.param_groups[1]['lr']))

    def update_lr(self, epoch, step):
        global_step = epoch * self.train_steps + step + 1
        global_step_without_warm_step = epoch * self.train_steps + step + 1 - self.warm_steps
        if global_step < self.warm_steps:
            self.optimizer.param_groups[0]['lr'] = self.lr[0] * global_step / self.warm_steps
            self.optimizer.param_groups[1]['lr'] = self.lr[1] * global_step / self.warm_steps
        elif global_step == self.warm_steps:
            self.optimizer.param_groups[0]['lr'] = self.lr[0]
            self.optimizer.param_groups[1]['lr'] = self.lr[1]
            print("small lr:{} large lr:{}".format(self.lr[0], self.lr[1]))
        elif global_step < self.train_steps and global_step_without_warm_step % 100 == 0:
            self.optimizer.param_groups[0]['lr'] = self.lr[0] * global_step_without_warm_step / (self.train_steps * self.epochs)
            self.optimizer.param_groups[1]['lr'] = self.lr[1] * global_step_without_warm_step / (self.train_steps * self.epochs)
        elif global_step % 10 == 0:
            cycle_step = self.train_steps * 4
            # lr_0 = (self.lr[0] - eta_min) * (1 - global_step / (self.train_steps * self.epochs)) + eta_min
            # lr_1 = (self.lr[1] - eta_min) * (1 - global_step / (self.train_steps * self.epochs)) + eta_min
            lr_0 = self.lr[0]
            lr_1 = self.lr[1]
            rate = np.cos(np.pi * global_step % cycle_step) + 1
            self.optimizer.param_groups[0]['lr'] = lr_0 * rate
            self.optimizer.param_groups[1]['lr'] = lr_1 * rate
        return self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[1]['lr']
