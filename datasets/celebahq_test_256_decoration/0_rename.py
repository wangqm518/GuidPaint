import os

def main():
    # 获得所有图片的路径
    image_paths=[]
    dir = os.getcwd()
    for f in os.listdir(dir):
        if f.endswith('.jpg'):
            image_paths.append(os.path.join(dir, f))
    for image_path in image_paths:
        image_name, ext = os.path.splitext(os.path.basename(image_path))
        new_image_name = f"{int(image_name):05d}{ext}"
        new_image_path = os.path.join(dir, new_image_name)
        os.rename(image_path, new_image_path)

if __name__ == '__main__':
    main()