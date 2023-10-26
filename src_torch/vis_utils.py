import matplotlib.pyplot as plt
import torch

def visualize_feature_maps(model, image, layer_name):
    # Set the model to evaluation mode
    torch.manual_seed(0)
    model.to('cpu')
    model.eval()

    feature_map_images = []

    # Define a hood to retrieve the feature maps from the specified layer
    def hook(module, input, output):
        nonlocal feature_map_images
        feature_maps = output
        if len(feature_maps.shape) == 4: # Check if it's a 4D tensor (e.g., convolutional layer)
            feature_map_images.append(feature_maps)
    
    layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            layer = module
            break
    
    if layer is None:
        print(f"Layer '{layer_name}' not found in the model")
        return

    handle = layer.register_forward_hook(hook)

    # Add the batch dimension to the image
    image = image.unsqueeze(0)

    with torch.no_grad():
        model(image)
    
    # Visualize the feature maps

    # Getting the tensor out of the list
    feature_maps = feature_map_images[0] # (1, C, H, W)
    feature_maps = feature_maps.squeeze(0) # (C, H, W)
    num_feature_maps = feature_maps.shape[0]

    features_per_row = 4
    if num_feature_maps < features_per_row:
        features_per_row = num_feature_maps

    num_rows = num_feature_maps // features_per_row
    if num_feature_maps % features_per_row != 0:
        num_rows += 1
    
    fig, axs = plt.subplots(num_rows, features_per_row, figsize=(10, 5))
    fig.suptitle(f'Feature Maps of Layer {layer_name}', fontsize=16)

    if num_feature_maps > 1:
        axs = axs.flatten()
        for i, ax, in enumerate(axs):
            if i >= num_feature_maps:
                break
            ax.set_title(f"Feature Map {i+1}")
            feature_map = feature_maps[i]
            ax.imshow(feature_map, cmap='gray')
            ax.axis('off')
    else:
        axs.set_title(f"Feature Map {1}")
        feature_map = feature_maps[0]
        axs.imshow(feature_map, cmap='gray')
        axs.axis('off')

    
    plt.show()