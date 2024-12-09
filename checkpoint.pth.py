# 체크 포인트를 통한 학습 중단 시 이어서 하는 코드 (checkpoint.pth)

import os
import torch

# 체크포인트 경로 설정
checkpoint_path = "/home/work/Dacon_Dataset/checkpoint.pth"

# 모델 및 옵티마이저 초기화
generator = UNet().to(device)
discriminator = PatchGANDiscriminator().to(device)

optimizer_G = optim.AdamW(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = optim.AdamW(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

# 체크포인트 로드 (있으면 이어서 학습)
start_epoch = 0
if os.path.exists(checkpoint_path):
    print(f"Checkpoint found at {checkpoint_path}. Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}.")
else:
    print("No checkpoint found. Starting training from scratch.")

# 데이터셋 및 DataLoader 정의
train_dataset = ImageDataset("/home/work/Dacon_Dataset/train_input", "/home/work/Dacon_Dataset/train_gt")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1, pin_memory=True)

epochs = 100
result_dir = "/home/work/Dacon_Dataset/result"
os.makedirs(result_dir, exist_ok=True)

# 학습 루프
for epoch in range(start_epoch, epochs):  # 중단된 지점부터 시작
    generator.train()
    discriminator.train()
    running_loss_G = 0.0
    running_loss_D = 0.0

    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        for input_images, gt_images in train_loader:
            input_images, gt_images = input_images.to(device), gt_images.to(device)

            real_labels = torch.ones_like(discriminator(gt_images)).to(device)
            fake_labels = torch.zeros_like(discriminator(input_images)).to(device)

            # Generator 학습
            optimizer_G.zero_grad()
            fake_images = generator(input_images)
            pred_fake = discriminator(fake_images)

            g_loss_adv = adversarial_loss(pred_fake, real_labels)
            g_loss_pixel = pixel_loss(fake_images, gt_images)
            g_loss = g_loss_adv + 100 * g_loss_pixel
            g_loss.backward()
            optimizer_G.step()

            # Discriminator 학습
            optimizer_D.zero_grad()
            pred_real = discriminator(gt_images)
            loss_real = adversarial_loss(pred_real, real_labels)

            pred_fake = discriminator(fake_images.detach())
            loss_fake = adversarial_loss(pred_fake, fake_labels)

            d_loss = (loss_real + loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            running_loss_G += g_loss.item()
            running_loss_D += d_loss.item()

            pbar.set_postfix(generator_loss=g_loss.item(), discriminator_loss=d_loss.item())
            pbar.update(1)

    print(f"Epoch [{epoch}/{epochs}] - Generator Loss: {running_loss_G / len(train_loader):.4f}, Discriminator Loss: {running_loss_D / len(train_loader):.4f}")

    # 테스트 및 출력 저장
    test_input_dir = "/home/work/Dacon_Dataset/test_input"
    output_dir = f"output_images_epoch_{epoch}"
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for img_name in sorted(os.listdir(test_input_dir)):
            img_path = os.path.join(test_input_dir, img_name)
            img = cv2.imread(img_path)
            input_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            output = generator(input_tensor).squeeze().permute(1, 2, 0).cpu().numpy() * 255.0
            output = output.astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, img_name), output)

    zip_filename = os.path.join(result_dir, f"epoch_{epoch}.zip")
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for img_name in os.listdir(output_dir):
            zipf.write(os.path.join(output_dir, img_name), arcname=img_name)
    print(f"Epoch {epoch} results saved to {zip_filename}")

    # 체크포인트 저장
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict()
    }, checkpoint_path)

generator.train()  
discriminator.train()
