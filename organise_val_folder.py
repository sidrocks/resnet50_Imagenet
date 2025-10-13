import os
import shutil
import xml.etree.ElementTree as ET

# Set the absolute paths to your 'val' folder and annotation folder
val_folder = '/workspace/era-misc-sandbox/Assignment_9/imagenet/ILSVRC/Data/CLS-LOC/val'  # Path to the val images
annotation_folder = '/workspace/era-misc-sandbox/Assignment_9/imagenet/ILSVRC/Annotations/CLS-LOC/val'  # Path to the annotation XML files

# Function to parse the XML annotation files and extract the class ID
def get_class_id_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # Assuming the class label is in the 'object' tag and 'name' sub-tag (this may vary based on the XML structure)
    class_id = root.find('.//name').text
    return class_id

# Loop through the XML annotation files and organize the images
for xml_file in os.listdir(annotation_folder):
    if xml_file.endswith('.xml'):
        # Get the corresponding image filename (same name as XML file)
        image_name = xml_file.replace('.xml', '.JPEG')
        image_path = os.path.join(val_folder, image_name)

        if os.path.exists(image_path):  # Make sure the image exists
            # Get the class ID from the XML file
            xml_path = os.path.join(annotation_folder, xml_file)
            class_id = get_class_id_from_xml(xml_path)

            # Create the class folder if it doesn't exist
            class_folder = os.path.join(val_folder, class_id)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

            # Move the image to the corresponding class folder
            dst = os.path.join(class_folder, image_name)
            shutil.move(image_path, dst)

print("Validation images organized successfully!")
