for indx,item in enumerate(range(100)):


    #to GPU
    torch.backends.cudnn.benchmark = True
    torch.cuda.synchronize()
    c = time.perf_counter()
    true_masks = dumin.to(device=device)  # item["seg"].to(device=device)
    imgs = dumin.to(device=device)   # item["data"].to(device=device)
    torch.cuda.synchronize()
    print(f"Transferring images: {time.perf_counter()-c:2.2f}")

    #Cost

    c = time.perf_counter()
    masks_pred = net(imgs)
    print(f"Forward: {time.perf_counter() - c:2.2f}")
    c = time.perf_counter()
    beloss = criterion(masks_pred, true_masks)
    torch.cuda.synchronize()
    print(f"Loss Calc: {time.perf_counter() - c:2.2f}")


    #backprop
    c = time.perf_counter()
    optimizer.zero_grad()
    beloss.backward()
    torch.cuda.synchronize()
    print(f"Back prop: {time.perf_counter() - c:2.2f}")

    #update
    torch.cuda.synchronize()
    c = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize()
    print(f"Step: {time.perf_counter() - c:2.2f}")