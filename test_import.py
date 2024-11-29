try:
    from sam2_train.modeling.backbones.image_encoder import ImageEncoder
    print("ImageEncoder imported successfully!")
except AttributeError as e:
    print(f"Error importing ImageEncoder: {e}")