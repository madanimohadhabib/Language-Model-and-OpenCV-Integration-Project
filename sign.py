from transformers import pipeline
import cv2
import numpy as np

# Load the text generation model
generator = pipeline('text-generation', model='gpt2')

# Input prompt for text generation
prompt = "Once in a distant future,"

# Generate text
results = generator(prompt, max_length=1000, num_return_sequences=1)

# Get the generated text
generated_text = results[0]['generated_text']

# Print the generated text to the console
print("Generated Text:", generated_text)

# Set font and calculate text size
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_color = (0, 0, 0)
line_type = 2

# Calculate the size of the text
text_size = cv2.getTextSize(generated_text, font, font_scale, line_type)[0]

# Set image dimensions based on text size
image_width = text_size[0] + 40  # Add some padding
image_height = text_size[1] + 40  # Add some padding

# Create an image with a white background
image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255

# Calculate the center position for the text
text_x = (image_width - text_size[0]) // 2
text_y = (image_height + text_size[1]) // 2

# Put the text on the image
cv2.putText(image, generated_text, (text_x, text_y), font, font_scale, font_color, line_type)

# Display the image
cv2.imshow('Generated Text Image', image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
