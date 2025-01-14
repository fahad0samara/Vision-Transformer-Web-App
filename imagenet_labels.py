"""Standard ImageNet class labels for image classification."""

IMAGENET_CLASSES = {
    # Animals
    0: 'tench',
    1: 'goldfish',
    2: 'great white shark',
    3: 'tiger shark',
    4: 'hammerhead shark',
    
    # Birds
    15: 'robin',
    16: 'bulbul',
    17: 'jay',
    18: 'magpie',
    19: 'chickadee',
    20: 'water ouzel',
    
    # Dogs
    151: 'chihuahua',
    152: 'japanese spaniel',
    153: 'maltese dog',
    154: 'pekinese',
    155: 'shih-tzu',
    156: 'king charles spaniel',
    157: 'papillon',
    
    # Cats
    281: 'tabby cat',
    282: 'tiger cat',
    283: 'persian cat',
    284: 'siamese cat',
    285: 'egyptian cat',
    
    # Electronics
    484: 'digital watch',
    515: 'digital clock',
    516: 'desktop computer',
    517: 'laptop',
    518: 'keyboard',
    519: 'mouse',
    520: 'trackball',
    521: 'webcam',
    522: 'scanner',
    743: 'printer',
    835: 'monitor',
    
    # Mobile Devices
    620: 'smartphone',
    621: 'tablet computer',
    622: 'e-reader',
    
    # Household Items
    530: 'table lamp',
    531: 'desk lamp',
    532: 'floor lamp',
    533: 'chandelier',
    534: 'vacuum',
    535: 'dishwasher',
    536: 'refrigerator',
    537: 'washing machine',
    538: 'microwave',
    539: 'toaster',
    
    # Furniture
    550: 'chair',
    551: 'armchair',
    552: 'sofa',
    553: 'bed',
    554: 'dining table',
    555: 'coffee table',
    556: 'desk',
    557: 'bookshelf',
    558: 'wardrobe',
    
    # Kitchen Items
    570: 'plate',
    571: 'bowl',
    572: 'cup',
    573: 'fork',
    574: 'knife',
    575: 'spoon',
    576: 'pot',
    577: 'pan',
    
    # Transportation
    650: 'car',
    651: 'bus',
    652: 'truck',
    653: 'motorcycle',
    654: 'bicycle',
    655: 'train',
    656: 'airplane',
    657: 'helicopter',
    
    # Nature
    970: 'coral reef',
    971: 'rock',
    972: 'mountain',
    973: 'volcano',
    974: 'tree',
    975: 'flower',
    976: 'grass',
    977: 'forest',
    978: 'beach',
    979: 'desert',
    
    # Previously added
    323: 'monarch butterfly',
    392: 'grasshopper',
    586: 'web browser',
    959: 'garden spider'
}

def get_class_label(class_id):
    """Get human-readable label for a class ID."""
    return IMAGENET_CLASSES.get(class_id, f'Unknown Class {class_id}')
