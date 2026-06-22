Segmented Keloid Dataset -- SLAAAAY

The Keloid folder contains two subfolders:
  - Images: Contains 30 keloid scar images sourced from the DermNet dataset: https://dermnetnz.org/images/keloid-and-hypertrophic-scar-images
  - Masks: Contains 30 manually annotated binary segmentation masks corresponding to the images in the same order, prepared for U-Net training.

The dataset annotations were originally provided by CVAT (Computer Vision Annotation Tool) in an annotations.xml file. Since CVAT's mask export features require a paid subscription lol, additional code was developed to convert the coordinate-based annotations into PNG segmentation masks compatible with U-Net training pipelines and preserving original resolution.

<img width="671" height="616" alt="image" src="https://github.com/user-attachments/assets/1ba60bc1-3659-41f3-b06f-7128b56cdfe9" />
