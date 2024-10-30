tools_in = [
    {
        'type': 'function',
        'function': {
            'name': "analyze_image",
            'description': "Working with the images.",
            'parameters': {
                'type': "object",
                'properties': {
                    'prompt': {
                        'type': "string",
                        'description': "Prompt for the LLM, which tells it what to do with the image.",
                    },
                },
            'required': ["prompt"],
        },
        }  
    },
    {
        'type': 'function',
        'function': {
            'name': "analyze_document",
            'description': "Working with the document files.",
            'parameters': {
                'type': "object",
                'properties': {
                    'prompt': {
                        'type': "string",
                        'description': "Prompt for the LLM, which tells it what to do with the document.",
                    },
                },
            'required': ["prompt"],
        },
        }  
    }
]

tools_out = [
    {
        'type': 'function',
        'function': {
            'name': "generate_image",
            'description': "Generate the image based on the prompt.",
            'parameters': {
                'type': "object",
                'properties': {
                    'prompt': {
                        'type': "string",
                        'description': "Prompt for image generation (description of the image).",
                    },
                    'filename': {
                        'type': "string",
                        'description': "Name of the generated image file with file extension. Format: {user_id}_{filename}",
                    },
                },
            'required': ["prompt", "filename"],
        },
        }  
    },
]


ratios = {
    "1:1": (1440, 1440),
    "4:3": (1600, 1200),
    "5:4": (1600, 1280),
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
}