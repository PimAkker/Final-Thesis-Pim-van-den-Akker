from lxml import etree

def get_namespace(element):
    # Extract namespace from the tag
    namespace = element.tag.split('}')[0].strip('{')
    return namespace

def change_svg_color_in_style(input_file, output_file, old_color, new_color):
    # Parse the SVG file
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(input_file, parser)
    root = tree.getroot()

    # Extract the namespace
    ns = {'svg': get_namespace(root)}

    # Find all elements with a style attribute
    for elem in root.xpath('//*[@style]', namespaces=ns):
        style = elem.get('style')
        # Split the style into individual declarations
        style_items = style.split(';')
        new_style_items = []
        for item in style_items:
            # Only modify the fill or stroke property if it matches the old color
            if old_color in item:
                new_style= item.replace(old_color, new_color)
                elem.set('style', new_style)

    # Save the modified SVG to a new file
    with open(output_file, 'wb') as f:
        tree.write(f, pretty_print=False, xml_declaration=True, encoding='UTF-8')

# Example usage
input_file = r'C:\Users\pimde\OneDrive\thesis\Blender\real_world_data\raw_csv\svg files\room1_5_2.svg'
output_file = r'C:\Users\pimde\OneDrive\thesis\Blender\real_world_data\raw_csv\svg files\test\modified_example.svg'
old_color = '#0000ff'  # Blue
new_color = '#00ff00'  # Green
import os
files = os.listdir(r'C:\Users\pimde\OneDrive\thesis\Blender\real_world_data\raw_csv\svg files')
files = [file for file in files if file.endswith('.svg')]

for file in files:
    input_file = r'C:\Users\pimde\OneDrive\thesis\Blender\real_world_data\raw_csv\svg files\{}'.format(file)
    output_file = r'C:\Users\pimde\OneDrive\thesis\Blender\real_world_data\raw_csv\svg_files_no_tables\{}'.format(file)
    change_svg_color_in_style(input_file, output_file, old_color, new_color)
