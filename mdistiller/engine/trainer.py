import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import getpass
from tensorboardX import SummaryWriter
from .utils import (
    AverageMeter,
    accuracy,
    validate,
    validate_view,
    validate_view_tekap,
    adjust_learning_rate,
    save_checkpoint,
    load_checkpoint,
    log_msg,
)
from .dot import DistillationOrientedTrainer


def check_model_gradients(model_instance, prefix=""):
    """
    주어진 모델 인스턴스 내의 모든 파라미터에 대해 requires_grad 상태와
    역전파 후 grad (기울기) 값을 확인합니다.
    """
    print(f"\n--- Checking Gradients for Model/Module: {prefix or model_instance.__class__.__name__} ---")

    has_trainable_params = False
    has_grad_values = False
    has_none_grads = False
    has_all_zeros_grads = False

    for name, param in model_instance.named_parameters():
        full_param_name = f"{prefix}.{name}" if prefix else name
        
        # 1. requires_grad 상태 확인
        if param.requires_grad:
            has_trainable_params = True
            
            # 2. grad 값 확인 (loss.backward() 호출 후)
            if param.grad is not None:
                has_grad_values = True
                grad_mean_abs = param.grad.abs().mean().item()
                grad_max_abs = param.grad.abs().max().item()
                
                # 모든 기울기가 0인지 확인 (Vanishing Gradient의 강력한 신호)
                if (param.grad == 0).all():
                    has_all_zeros_grads = True
                    print(f"  [WARN] Parameter: {full_param_name}, requires_grad=True, GRAD IS ALL ZEROS! (Mean Abs: {grad_mean_abs:.8f}, Max Abs: {grad_max_abs:.8f})")
                elif grad_mean_abs < 1e-8 and grad_max_abs < 1e-6: # 매우 작은 기울기
                     print(f"  [WARN] Parameter: {full_param_name}, requires_grad=True, GRAD IS VERY SMALL! (Mean Abs: {grad_mean_abs:.8f}, Max Abs: {grad_max_abs:.8f})")
                else:
                    print(f"  [INFO] Parameter: {full_param_name}, requires_grad=True, Grad (Mean Abs: {grad_mean_abs:.8f}, Max Abs: {grad_max_abs:.8f})")
            else:
                has_none_grads = True
                print(f"  [ERROR] Parameter: {full_param_name}, requires_grad=True, GRAD IS NONE! (Gradient flow broken?)")
        else:
            # requires_grad가 False인 파라미터는 기울기가 계산되지 않으므로 grad=None이 정상
            if param.grad is None:
                print(f"  [INFO] Parameter: {full_param_name}, requires_grad=False (Expected grad=None)")
            else:
                # requires_grad가 False인데 grad가 None이 아니라면 비정상
                print(f"  [WARN] Parameter: {full_param_name}, requires_grad=False, BUT GRAD IS NOT NONE! (Might be leftover from previous run or anomaly)")

    if not has_trainable_params:
        print("  No trainable parameters found in this module.")
    if has_none_grads:
        print("\n  >>> SUMMARY: Some trainable parameters had None gradients! Check computational graph and detach() calls. <<<")
    if has_all_zeros_grads:
        print("\n  >>> SUMMARY: Some trainable parameters had all-zero gradients! Check for dead ReLUs or vanishing gradients. <<<")
    if not has_grad_values and has_trainable_params:
        print("\n  >>> SUMMARY: No gradient values found for trainable parameters! Something is critically wrong. <<<")
    print("--------------------------------------------------")

class BaseTrainer(object):
    def __init__(self, experiment_name, distiller, train_loader, val_loader, cfg):
        self.cfg = cfg
        self.distiller = distiller
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = self.init_optimizer(cfg)
        self.best_acc = -1

        username = getpass.getuser()
        # init loggers
        self.log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.tf_writer = SummaryWriter(os.path.join(self.log_path, "train.events"))

    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            optimizer = optim.SGD(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def log(self, lr, epoch, log_dict):
        # tensorboard log
        for k, v in log_dict.items():
            self.tf_writer.add_scalar(k, v, epoch)
        self.tf_writer.flush()
        # wandb log
        if self.cfg.LOG.WANDB:
            import wandb

            wandb.log({"current lr": lr})
            wandb.log(log_dict)
        if log_dict["test_acc"] > self.best_acc:
            self.best_acc = log_dict["test_acc"]
            if self.cfg.LOG.WANDB:
                wandb.run.summary["best_acc"] = self.best_acc
        # worklog.txt
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            lines = [
                "-" * 25 + os.linesep,
                "epoch: {}".format(epoch) + os.linesep,
                "lr: {:.2f}".format(float(lr)) + os.linesep,
            ]
            for k, v in log_dict.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                lines.append("{}: {:.2f}".format(k, v) + os.linesep)
            lines.append("-" * 25 + os.linesep)
            writer.writelines(lines)

    def train(self, resume=False):
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1
        print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))

    def train_epoch(self, epoch):
        lr = adjust_learning_rate(epoch, self.cfg, self.optimizer)
        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "loss_student_target": AverageMeter(),
            "loss_student_teacher_logit": AverageMeter(),
            "loss_student_teacher_feat": AverageMeter(),
            "loss_student_total": AverageMeter(),
            'loss_teacher_target': AverageMeter(),
            "loss_ensemble" : AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }
        num_iter = len(self.train_loader)
        pbar = tqdm(range(num_iter))

        # train loops
        self.distiller.train()
        if self.cfg.DIV.USAGE:
            self.distiller.module.teacher.view_generator.train()
            
        for idx, data in enumerate(self.train_loader):
            msg = self.train_iter(data, epoch, train_meters)
            pbar.set_description(log_msg(msg, "TRAIN"))
            pbar.update()
        pbar.close()
        
        # validate
        test_acc, test_acc_top5, test_loss = validate(self.val_loader, self.distiller)

        # log
        log_dict = OrderedDict(
            {
                "train_acc": train_meters["top1"].avg,
                "train_loss": train_meters["losses"].avg,
                "test_acc": test_acc,
                "test_acc_top5": test_acc_top5,
                "test_loss": test_loss,
            }
        )
        
        if self.cfg.DIV.USAGE:
            teacher_acc, view_acc_list = validate_view(self.val_loader, self.distiller)
            print("View ACC ",view_acc_list)
            metric = {}
            for k in train_meters.keys():
                metric[k] = train_meters[k].avg                
            print("Loss", metric)
            
            log_dict['teacher_acc'] = teacher_acc
            log_dict['view1_acc'] = view_acc_list[0]
            log_dict['view2_acc'] = view_acc_list[1]
            log_dict['view3_acc'] = view_acc_list[2]
            log_dict['view4_acc'] = view_acc_list[3]
            log_dict['view5_acc'] = view_acc_list[4]
            
            log_dict["loss_student_target"] = train_meters["loss_student_target"].avg
            log_dict["loss_student_teacher_logit"] = train_meters["loss_student_teacher_logit"].avg
            log_dict["loss_student_teacher_feat"] = train_meters["loss_student_teacher_feat"].avg
            log_dict["loss_student_total"] = train_meters["loss_student_total"].avg
            log_dict["loss_teacher_target"] = train_meters['loss_teacher_target'].avg
            log_dict['loss_ensemble'] = train_meters['loss_ensemble'].avg
            
        # if self.cfg.TEKAP.USAGE:
        #     teacher_acc, view_acc_list = validate_view_tekap(self.val_loader, self.distiller)
        #     print("View ACC ",view_acc_list)
        #     for k in train_meters.keys():
        #         metric[k] = train_meters[k].avg                
        #     print("Loss", metric)
            
        #     log_dict['teacher_acc'] = teacher_acc
        #     log_dict['view1_acc'] = view_acc_list[0]
        #     log_dict['view2_acc'] = view_acc_list[1]
        #     log_dict['view3_acc'] = view_acc_list[2]
        #     log_dict['view4_acc'] = view_acc_list[3]
        #     log_dict['view5_acc'] = view_acc_list[4]
            
        #     log_dict["loss_student_target"] = train_meters["loss_student_target"].avg
        #     log_dict["loss_student_teacher_logit"] = train_meters["loss_student_teacher_logit"].avg
        #     log_dict["loss_student_teacher_feat"] = train_meters["loss_student_teacher_feat"].avg
        #     log_dict["loss_student_total"] = train_meters["loss_student_total"].avg
        #     log_dict["loss_teacher_target"] = train_meters['loss_teacher_target'].avg
            
        self.log(lr, epoch, log_dict)
        # saving checkpoint
        state = {
            "epoch": epoch,
            "model": self.distiller.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
        }
        student_state = {"model": self.distiller.module.student.state_dict()}
        save_checkpoint(state, os.path.join(self.log_path, "latest"))
        save_checkpoint(
            student_state, os.path.join(self.log_path, "student_latest")
        )
        if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
            save_checkpoint(
                state, os.path.join(self.log_path, "epoch_{}".format(epoch))
            )
            save_checkpoint(
                student_state,
                os.path.join(self.log_path, "student_{}".format(epoch)),
            )
        # update the best
        if test_acc >= self.best_acc:
            save_checkpoint(state, os.path.join(self.log_path, "best"))
            save_checkpoint(
                student_state, os.path.join(self.log_path, "student_best")
            )

    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(image=image, target=target, epoch=epoch)

        # backward                                         
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class CRDTrainer(BaseTrainer):
    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index, contrastive_index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        contrastive_index = contrastive_index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(
            image=image, target=target, index=index, contrastive_index=contrastive_index, epoch=epoch
        )
        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        
        train_meters['loss_teacher_target'].update(losses_dict['ce_loss'].cpu().detach().numpy().mean(), batch_size)
        train_meters['loss_student_target'].update(losses_dict['loss_student_target'].cpu().detach().numpy().mean(), batch_size)
        train_meters['loss_student_teacher_logit'].update(losses_dict['loss_student_teacher_logit'].cpu().detach().numpy().mean(), batch_size)
        train_meters['loss_student_teacher_feat'].update(losses_dict['loss_student_teacher_feat'].cpu().detach().numpy().mean(), batch_size)
        
        student_loss = sum([losses_dict['loss_student_target'].mean(),  losses_dict['loss_student_teacher_logit'].mean(),losses_dict['loss_student_teacher_feat'].mean()])
        train_meters['loss_student_total'].update(student_loss.cpu().detach().numpy().mean(), batch_size)
        
        feat_inter_loss = losses_dict['feature_inter_loss'].mean()
        feat_intra_loss = losses_dict['feature_intra_loss'].mean()
        logit_inter_loss = losses_dict['logit_inter_loss'].mean()
        logit_intra_loss = losses_dict['logit_intra_loss'].mean()
        
        train_meters['loss_ensemble'].update(sum([feat_inter_loss, feat_intra_loss, logit_inter_loss, logit_intra_loss]))
        
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class DOT(BaseTrainer):
    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            m_task = cfg.SOLVER.MOMENTUM - cfg.SOLVER.DOT.DELTA
            m_kd = cfg.SOLVER.MOMENTUM + cfg.SOLVER.DOT.DELTA
            optimizer = DistillationOrientedTrainer(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=m_task,
                momentum_kd=m_kd,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def train(self, resume=False):
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1
        print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))

    def train_iter(self, data, epoch, train_meters):
        train_start_time = time.time()
        image, target, index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(image=image, target=target, epoch=epoch)

        # dot backward
        loss_ce, loss_kd = losses_dict['loss_ce'].mean(), losses_dict['loss_kd'].mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss_kd.backward(retain_graph=True)
        self.optimizer.step_kd()
        self.optimizer.zero_grad(set_to_none=True)
        loss_ce.backward()
        self.optimizer.step()

        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update((loss_ce + loss_kd).cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class CRDDOT(BaseTrainer):

    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            m_task = cfg.SOLVER.MOMENTUM - cfg.SOLVER.DOT.DELTA
            m_kd = cfg.SOLVER.MOMENTUM + cfg.SOLVER.DOT.DELTA
            optimizer = DistillationOrientedTrainer(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=m_task,
                momentum_kd=m_kd,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def train(self, resume=False):
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1
        print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))

    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index, contrastive_index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        contrastive_index = contrastive_index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(
            image=image, target=target, index=index, contrastive_index=contrastive_index,  epoch=epoch
        )

        # dot backward
        
        for k in losses_dict:
            losses_dict[k] = losses_dict[k].mean()

        loss_ce = losses_dict.get('loss_ce', 0.0)
        loss_kd = losses_dict.get('loss_kd', 0.0)

        for k, v in losses_dict.items():
            if k not in ['loss_ce', 'loss_kd']:
                loss_kd += v
                
        losses_dict = {
            'loss_ce': loss_ce,
            'loss_kd': loss_kd
        }
                
            
        #loss_ce, loss_kd = losses_dict['loss_ce'].mean(), losses_dict['loss_kd'].mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss_kd.backward(retain_graph=True)
        self.optimizer.step_kd()
        self.optimizer.zero_grad(set_to_none=True)
        loss_ce.backward()
        # self.optimizer.step((1 - epoch / 240.))
        self.optimizer.step()

        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update((loss_ce + loss_kd).cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class CRDAUGTrainer(BaseTrainer):    
    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index, contrastive_index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image_weak, image_strong = image
        image_weak, image_strong = image_weak.float(), image_strong.float()
        image_weak, image_strong = image_weak.cuda(non_blocking=True), image_strong.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(image_weak=image_weak, image_strong=image_strong, target=target, index=index, contrastive_index=contrastive_index, epoch=epoch)

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image_weak.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg