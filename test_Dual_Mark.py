import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import *
from network.Dual_Mark import *
import os
import time
from shutil import copyfile
from network.noise_layers import *
from PIL import Image
import random, string
import os
from torchvision import transforms
import lpips
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

'''
test
'''
criterion_LPIPS = lpips.LPIPS().to("cuda")

def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_path(path="temp/"):
    return path + ''.join(random.sample(string.ascii_letters + string.digits, 16)) + ".png"


def main():
    os.makedirs('temp', exist_ok=True)
    seed_torch(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('cfg/test_DualMark.yaml', 'r') as f:
        args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    result_folder = "results/" + args.result_folder
    model_epoch = args.model_epoch
    batch_size = args.batch_size
    strength_factor = args.strength_factor
    with_mask = args.with_mask
    noise_layer = args.noise_layer

    with open(result_folder + '/train_DualMark.yaml', 'r') as f:
        args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    lr = args.lr
    beta1 = args.beta1
    image_size = args.image_size
    message_length = args.message_length
    message_range = args.message_range
    attention_encoder = args.attention_encoder
    attention_decoder = args.attention_decoder
    weight = args.weight
    dataset_path = args.dataset_path
    save_images_number = args.save_images_number
    noise_layers_R = args.noise_layers.pool_R
    noise_layers_F = args.noise_layers.pool_F

    test_log = result_folder + "test_log" + time.strftime("%_Y_%m_%d__%H_%M_%S", time.localtime()) + ".txt"
    copyfile("cfg/test_DualMark.yaml", result_folder + "test_DualMark" + time.strftime("_%Y_%m_%d__%H_%M_%S", time.localtime()) + ".yaml")
    writer = SummaryWriter('runs/' + result_folder + noise_layer + time.strftime("%_Y_%m_%d__%H_%M_%S", time.localtime()))

    network = Network(message_length, noise_layers_R, noise_layers_F, device, batch_size, lr, beta1, attention_encoder, attention_decoder, weight)
    EC_path = result_folder + "models/EC_" + str(model_epoch) + ".pth"
    network.load_model_ed(EC_path)

    if noise_layer[0:len("StarGAN")] != "StarGAN":
        test_dataset = maskImgDataset(dataset_path, image_size)
    else:
        test_dataset = attrsImgDataset(dataset_path, image_size, "celebahq")

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print("\nStart Testing : \n\n")

    test_result = {
        "error_rate_C": 0.0,
        "error_rate_RF": 0.0,
        "bit_accuracy_C": 0.0,
        "bit_accuracy_RF": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "lpips": 0.0,
        "embed_time": 0.0,
        "decode_time_C": 0.0,
        "decode_time_RF": 0.0
    }

    saved_iterations = np.random.choice(np.arange(1, len(test_dataloader)+1), size=save_images_number, replace=False)
    saved_all = None
    
    # Counter for total number of images processed
    total_images = 0

    print("Start for looop ==========")
    for step, (image, mask) in enumerate(test_dataloader, 1):
        image = image.to(device)
        message = torch.Tensor(np.random.choice([-message_range, message_range], (image.shape[0], message_length))).to(device)

        '''
        test
        '''
        network.encoder_decoder.eval()
        network.discriminator.eval()

        with torch.no_grad():
            # use device to compute
            images, messages, masks = image.to(network.device), message.to(network.device), mask.to(network.device)

            # Measure embedding time
            embed_start_time = time.time()
            encoded_images = network.encoder_decoder.module.encoder(images, messages)
            encoded_images = images + (encoded_images - images) * strength_factor
            torch.cuda.synchronize()  # Ensure GPU operations are complete
            embed_end_time = time.time()
            embed_time_batch = embed_end_time - embed_start_time
            embed_time_per_image = embed_time_batch / images.shape[0]

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            ##################################################################
            for index in range(encoded_images.shape[0]):
                single_image = ((encoded_images[index].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0,255).to('cpu', torch.uint8).numpy()
                im = Image.fromarray(single_image)
                file = get_path()
                while os.path.exists(file):
                    file = get_path()
                im.save(file)
                read = np.array(Image.open(file), dtype=np.uint8)
                os.remove(file)

                encoded_images[index] = transform(read).unsqueeze(0).to(image.device)
            ##################################################################
            # psnr
            psnr = - kornia.losses.psnr_loss(encoded_images.detach(), images, 2).item()

            # ssim
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=11, reduction="mean").item()

            # lpips
            lpips = torch.mean(criterion_LPIPS(encoded_images.detach(), images)).item()

            #noised_images_C, noised_images_R, noised_images_F = network.encoder_decoder.module.noise([encoded_images, images, masks])
            #network.encoder_decoder.module.noise.train()
            #_, _, noised_images = network.encoder_decoder.module.noise([encoded_images, images, masks])
            noised_images = eval(noise_layer)([encoded_images.clone(), images, masks])
            #noised_images = eval('JpegTest()')([noised_images.clone(), images, masks])

            ##################################################################
            for index in range(noised_images.shape[0]):
                single_image = ((noised_images[index].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0,255).to('cpu', torch.uint8).numpy()
                im = Image.fromarray(single_image)
                file = get_path()
                while os.path.exists(file):
                    file = get_path()
                im.save(file)
                read = np.array(Image.open(file), dtype=np.uint8)
                os.remove(file)

                noised_images[index] = transform(read).unsqueeze(0).to(image.device)
            ##################################################################

            # Measure decoding time for decoder C
            decode_start_time_C = time.time()
            decoded_messages_C = network.encoder_decoder.module.decoder_C(noised_images)
            torch.cuda.synchronize()  # Ensure GPU operations are complete
            decode_end_time_C = time.time()
            decode_time_C_batch = decode_end_time_C - decode_start_time_C
            decode_time_C_per_image = decode_time_C_batch / images.shape[0]

            # Measure decoding time for decoder RF
            decode_start_time_RF = time.time()
            decoded_messages_RF = network.encoder_decoder.module.decoder_RF(noised_images)
            torch.cuda.synchronize()  # Ensure GPU operations are complete
            decode_end_time_RF = time.time()
            decode_time_RF_batch = decode_end_time_RF - decode_start_time_RF
            decode_time_RF_per_image = decode_time_RF_batch / images.shape[0]

        '''
        decoded message error rate
        '''
        error_rate_C = network.decoded_message_error_rate_batch(messages, decoded_messages_C)
        error_rate_RF = network.decoded_message_error_rate_batch(messages, decoded_messages_RF)

        # Calculate bit accuracy (1 - error_rate)
        bit_accuracy_C = 1.0 - error_rate_C
        bit_accuracy_RF = 1.0 - error_rate_RF

        result = {
                "error_rate_C": error_rate_C,
                "error_rate_RF": error_rate_RF,
                "bit_accuracy_C": bit_accuracy_C,
                "bit_accuracy_RF": bit_accuracy_RF,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips,
                "embed_time": embed_time_per_image,
                "decode_time_C": decode_time_C_per_image,
                "decode_time_RF": decode_time_RF_per_image
            }

        # Accumulate results (multiply by batch size to get total time for averaging later)
        for key in result:
            if "time" in key:
                test_result[key] += float(result[key]) * images.shape[0]
            else:
                test_result[key] += float(result[key])

        # Update total image count
        total_images += images.shape[0]

        if step in saved_iterations:
            if saved_all is None:
                saved_all = get_random_images(image, encoded_images, noised_images)
            else:
                saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)

        '''
        test results
        '''
        content = "Batch " + str(step) + f" ({images.shape[0]} images) : \n"
        for key in result:
            if "time" in key:
                content += key + "=" + str(f"{result[key]:.8f}") + "s/image,"
            else:
                content += key + "=" + str(f"{result[key]:.8f}") + ","
            writer.add_scalar("Test/" + key, float(result[key]), step)
        content += "\n"

        with open(test_log, "a") as file:
            file.write(content)

        print(content)

    '''
    test results
    '''
    content = "Average per image : \n"
    for key in test_result:
        if "time" in key:
            avg_value = test_result[key] / total_images
            content += key + "=" + str(f"{avg_value:.8f}") + "s/image,"
        else:
            avg_value = test_result[key] / step
            content += key + "=" + str(f"{avg_value:.8f}") + ","
        writer.add_scalar("Test_epoch/" + key, float(avg_value), 1)
    content += "\n"
    
    # Additional timing statistics per image
    avg_embed_time = test_result["embed_time"] / total_images
    avg_decode_time_C = test_result["decode_time_C"] / total_images
    avg_decode_time_RF = test_result["decode_time_RF"] / total_images
    
    timing_summary = f"\n=== TIMING SUMMARY (PER IMAGE) ===\n"
    timing_summary += f"Total images processed: {total_images}\n"
    timing_summary += f"Average Embedding Time: {avg_embed_time:.8f}s per image\n"
    timing_summary += f"Average Decoding Time (C): {avg_decode_time_C:.8f}s per image\n"
    timing_summary += f"Average Decoding Time (RF): {avg_decode_time_RF:.8f}s per image\n"
    timing_summary += f"Total Average Processing Time: {avg_embed_time + avg_decode_time_C + avg_decode_time_RF:.8f}s per image\n"
    timing_summary += f"Images per Second (Embedding): {1.0 / avg_embed_time:.5f}\n"
    timing_summary += f"Images per Second (Decoding C): {1.0 / avg_decode_time_C:.5f}\n"
    timing_summary += f"Images per Second (Decoding RF): {1.0 / avg_decode_time_RF:.5f}\n"
    timing_summary += f"Total Images per Second: {1.0 / (avg_embed_time + avg_decode_time_C + avg_decode_time_RF):.5f}\n"
    timing_summary += "========================================\n"
    
    print("KEt qua cuoi cung:", content)
    print(timing_summary)
    
    with open(test_log, "a") as file:
        file.write(content)
        file.write(timing_summary)

    print(content)
    # save_images(saved_all, "test", result_folder + "images/", resize_to=None)

    writer.close()


if __name__ == '__main__':
    main()