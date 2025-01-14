import os
from image_processor import ImageProcessor

def main():
    print("Vision Transformer (ViT) Demo")
    print("-" * 50)
    
    # Initialize the image processor
    processor = ImageProcessor()
    
    # Create a samples directory if it doesn't exist
    samples_dir = "samples"
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
        print(f"Created {samples_dir} directory. Please add some images (.jpg, .png) to this directory.")
        print("Then run this script again.")
        return
    
    # Get all image files from the samples directory
    image_files = [f for f in os.listdir(samples_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {samples_dir} directory.")
        print("Please add some images (.jpg, .png) to this directory and run again.")
        return
    
    print(f"Found {len(image_files)} images to analyze.")
    print("-" * 50)
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(samples_dir, image_file)
        print(f"\nAnalyzing {image_file}...")
        
        try:
            results = processor.analyze_image(image_path)
            
            print("\nResults:")
            print("-" * 20)
            print("Top 5 Classifications:")
            for i, result in enumerate(results['classification_results'], 1):
                print(f"{i}. Class {result['class_id']} - Confidence: {result['confidence']:.4f}")
            
            print("\nMetadata:")
            for key, value in results['metadata'].items():
                print(f"{key}: {value}")
                
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
