def train(model, criterion, optimizer, scheduler, epochs, train_dataloader, val_dataloader, device):
    for epoch in range(epochs):
        model.train()
        loss_meter = AverageMeter("train_loss", ":.5f")
        epoch_outs, epoch_tgt = list(), list()
        for data, tgt_vec in tqdm(train_dataloader):
            data, tgt_vec = data.to(device), tgt_vec.to(device)
            targets = torch.argmax(tgt_vec, axis=1)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, targets)
            loss_meter.update(loss.item(), data.shape[0])
            loss.backward()
            optimizer.step()
            epoch_outs.append(out)
            epoch_tgt.append(tgt_vec)
        predictions = torch.vstack([torch.softmax(out, axis=1) for out in epoch_outs]).detach().cpu().numpy()
        targets = torch.cat([tgt for tgt in epoch_tgt], dim=0).detach().cpu().numpy()
        ap_score = average_precision_score(targets, predictions)
        eval_loss_meter, eval_ap_score = evaluate(model, criterion, val_dataloader, device)
        data_to_log = {
            "epoch": epoch+1,
            "train_loss": loss_meter.avg,
            "eval_loss": eval_loss_meter.avg,
            "train_ap_score": ap_score,
            "eval_ap_score": eval_ap_score,
            "lr": optimizer.state_dict()["param_groups"][0]["lr"],
        }
        scheduler.step(eval_loss_meter.avg)
        wandb.log(data_to_log)
        print(data_to_log)