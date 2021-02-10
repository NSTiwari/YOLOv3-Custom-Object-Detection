# YOLOv3-Custom-Object-Detection

## Steps:

1. Prepare your dataset and label them in YOLO format using `LabelImg`. Once done, zip all the images and their corresponding label files as `images.zip`.
2. Create a folder named `yolov3` on Drive and upload the `images.zip` file inside it. The directory structure should look something like the following:
```
yolov3
|__images.zip
   |__ *.jpg (all image files)
   |__ *.txt (all annotation files)
```
3. Clone the repository and upload the `YOLOv3_Custom_Object_Detection.ipynb` notebook on Google Colab.
4. Run the cells one-by-one by following instructions as stated in the notebook. For detailed explanation, refer the following [document]: https://github.com/NSTiwari/YOLOv3-Custom-Object-Detection/blob/main/YOLOv3%20Custom%20Object%20Detection%20with%20Transfer%20Learning.pdf
5. Once the training is completed, download the following files from the `yolov3` folder saved on Google Drive on your local machine.
   - `yolov3_training_last.weights`
   - `yolov3_testing.cfg`
   - `classes.txt`
6. Copy the downloaded files and save them inside the repository you had cloned on your local machine.
7. Create a new folder named `test_images` inside the repository and save some test images inside it which you'd like to test the model on.
8. Open the `Object_Detection.py` file inside the repository and edit **Line 17**  by replacing `<your_test_image` with the name of the image file you want to the test.
