bce_loss = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

dataloader = DataLoader(dataset_train, batch_size=256, shuffle=True, num_workers=2)

for _ in range(50):

    for batch in dataloader:

        optimizer.zero_grad()

        # meta_cats_batch = multi-label categories
        image_batch, tokens_batch, meta_cats_batch = batch
        inputs = (image_batch.to(DEVICE), tokens_batch.to(DEVICE), meta_cats_batch.to(DEVICE))

        with torch.set_grad_enabled(True):
            image_embeddings, text_embeddings, meta_preds = model(inputs)

        cosine_matrix = train.cosine_distance(image_embeddings, text_embeddings)

        mbmr_loss = train.calculate_loss(cosine_matrix, temperature=0.025)
        mbmr_loss /= image_batch.size(0)
        meta_cats_loss = Variable(bce_loss(meta_preds.detach().cpu(), meta_cats_batch), requires_grad=True)

        loss = mbmr_loss + meta_cats_loss
        loss.backward()
        optimizer.step()

        print('Meta Category BCE Loss:', meta_cats_loss.item())