import torch
from .utils import MetricLogger, multi_acc

def train_one_epoch(model, 
                    data_loader, 
                    criterion,
                    criterion2,
                    epoch, 
                    n_epochs,
                    log_freq,
                    fp16_scaler, 
                    optimizer,
                    scheduler):
    
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'TRAINING > Epoch: [{}/{}]'.format(epoch, n_epochs)
    n_batch = len(data_loader)
    for it, data in enumerate(metric_logger.log_every(data_loader, log_freq, header)):
        bscan_ti, bscan_tj = map(lambda f: f.cuda(non_blocking=True).type(torch.float), data[:2])
        side_eye, bscan_num, sex, age, delta_h= map(lambda f: f.cuda(non_blocking=True).type(torch.float), data[2:7])
        labels = data[7].cuda(non_blocking=True).type(torch.int64)

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            logits = model(bscan_ti, bscan_tj, side_eye, bscan_num, sex, age, delta_h)
            loss1 = criterion(logits, labels)
            loss2 = criterion2(logits, labels) * 0.08
            loss = loss1 + loss2 

        optimizer.zero_grad(set_to_none=True)
        fp16_scaler.scale(loss).backward()
        fp16_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., norm_type=2)
        fp16_scaler.step(optimizer)
        fp16_scaler.update()
        if scheduler is not None:
            scheduler.step((epoch*n_batch)+it)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss1=loss1.item())
        metric_logger.update(loss2=loss2.item())
        metric_logger.update(accuracy = multi_acc(logits, labels))
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("TRAINING > Averaged stats:", metric_logger)
    return {k: meter.avg for k, meter in metric_logger.meters.items()}


def valid_one_epoch(model, 
                    data_loader, 
                    criterion,
                    criterion2,
                    epoch, 
                    n_epochs, 
                    log_freq, 
                    fp16_scaler):
    
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header_val = 'Validation > Epoch: [{}/{}]'.format(epoch, n_epochs)

    with torch.no_grad():
        for _, data in enumerate(metric_logger.log_every(data_loader, log_freq, header_val)):
            bscan_ti, bscan_tj = map(lambda f: f.cuda(non_blocking=True).type(torch.float), data[:2])
            side_eye, bscan_num, sex, age, delta_h= map(lambda f: f.cuda(non_blocking=True).type(torch.float), data[2:7])
            labels = data[7].cuda(non_blocking=True).type(torch.int64)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                logits = model(bscan_ti, bscan_tj, side_eye, bscan_num, sex, age, delta_h)
                loss1 = criterion(logits, labels)
                loss2 = criterion2(logits, labels) * 0.08
                loss = loss1 + loss2 

            torch.cuda.synchronize()

            metric_logger.update(loss=loss.item())
            metric_logger.update(loss1=loss1.item())
            metric_logger.update(loss2=loss2.item())
            metric_logger.update(accuracy = multi_acc(logits, labels))

    metric_logger.synchronize_between_processes()
    print("Validation > Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_distill_one_epoch(teacher, 
                    student, 
                    data_loader,
                    ce_criterion,
                    focal_criterion,
                    cosine_loss,
                    epoch, 
                    n_epochs, 
                    log_freq, 
                    fp16_scaler,
                    optimizer,
                    scheduler,
                    T=2,
                    focal_loss_weight=0.6,
                    ce_loss_weight=0.1, 
                    cosine_loss_weight = 0.3
                    ):

    teacher.eval()
    student.train()
    metric_logger = MetricLogger(delimiter="  ")
    header_val = 'Trainin distill > Epoch: [{}/{}]'.format(epoch, n_epochs)
    n_batch = len(data_loader)
    for it, data in enumerate(metric_logger.log_every(data_loader, log_freq, header_val)):
        bscan_ti, bscan_tj = map(lambda f: f.cuda(non_blocking=True).type(torch.float), data[:2])
        side_eye, bscan_num, sex, age, delta_h= map(lambda f: f.cuda(non_blocking=True).type(torch.float), data[2:7])
        labels = data[7].cuda(non_blocking=True).type(torch.int64)
        localizer = data[8].cuda(non_blocking=True).type(torch.float)

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            with torch.no_grad():
                _, hidden_feat_teacher  = teacher(bscan_ti, bscan_tj, side_eye, bscan_num, sex, age, delta_h)
            logits_student, hidden_feat_student = student(bscan_ti, side_eye, bscan_num, sex, age, delta_h, localizer)

            hidden_rep_loss = cosine_loss(hidden_feat_student, hidden_feat_teacher, target=torch.ones(bscan_ti.size(0)).cuda(non_blocking=True))
            label_loss = ce_criterion(logits_student, labels) * 0.3
            focal_loss = focal_criterion(logits_student, labels) * 10

            loss = focal_loss_weight * focal_loss + ce_loss_weight * label_loss + hidden_rep_loss * cosine_loss_weight

        optimizer.zero_grad(set_to_none=True)
        fp16_scaler.scale(loss).backward()
        fp16_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1., norm_type=2)
        fp16_scaler.step(optimizer)
        fp16_scaler.update()
        scheduler.step((epoch*n_batch)+it)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss.item())
        metric_logger.update(focal_loss=focal_loss.item())
        metric_logger.update(hidden_loss=hidden_rep_loss.item())
        metric_logger.update(label_loss=label_loss.item())
        metric_logger.update(accuracy = multi_acc(logits_student, labels))
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("TRAINING > Averaged stats:", metric_logger)
    return {k: meter.avg for k, meter in metric_logger.meters.items()}


def valid_distill_one_epoch(teacher, 
                    student, 
                    data_loader,
                    ce_criterion,
                    focal_criterion,
                    cosine_loss,
                    epoch, 
                    n_epochs, 
                    log_freq, 
                    fp16_scaler,
                    T=2,
                    focal_loss_weight=0.6,
                    ce_loss_weight=0.1, 
                    cosine_loss_weight = 0.3
                    ):

    teacher.eval()
    student.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header_val = 'Trainin distill > Epoch: [{}/{}]'.format(epoch, n_epochs)

    with torch.no_grad():
        for it, data in enumerate(metric_logger.log_every(data_loader, log_freq, header_val)):
            bscan_ti, bscan_tj = map(lambda f: f.cuda(non_blocking=True).type(torch.float), data[:2])
            side_eye, bscan_num, sex, age, delta_h= map(lambda f: f.cuda(non_blocking=True).type(torch.float), data[2:7])
            labels = data[7].cuda(non_blocking=True).type(torch.int64)
            localizer = data[8].cuda(non_blocking=True).type(torch.float)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                _, hidden_feat_teacher  = teacher(bscan_ti, bscan_tj, side_eye, bscan_num, sex, age, delta_h)
                logits_student, hidden_feat_student = student(bscan_ti, side_eye, bscan_num, sex, age, delta_h, localizer)

                hidden_rep_loss = cosine_loss(hidden_feat_student, hidden_feat_teacher, target=torch.ones(bscan_ti.size(0)).cuda(non_blocking=True))
                label_loss = ce_criterion(logits_student, labels) * 0.3
                focal_loss = focal_criterion(logits_student, labels) * 10

                loss = focal_loss_weight * focal_loss + ce_loss_weight * label_loss + hidden_rep_loss * cosine_loss_weight

            torch.cuda.synchronize()

            metric_logger.update(loss=loss.item())
            metric_logger.update(focal_loss=focal_loss.item())
            metric_logger.update(hidden_loss=hidden_rep_loss.item())
            metric_logger.update(label_loss=label_loss.item())
            metric_logger.update(accuracy = multi_acc(logits_student, labels))

    metric_logger.synchronize_between_processes()
    print("Validation > Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}