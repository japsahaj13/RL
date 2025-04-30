#!/usr/bin/env python
import os
import re

def cleanup_config_file(file_path):
    """Remove specific legacy parameters from YAML configuration files."""
    # Parameters to remove
    legacy_params = [
        'competitor_effect:',
        'price_sensitivity:',
        'promo_boost:',
        'seasonal_amplitude:'
    ]
    
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Track which parameters were found and removed
    removed_params = []
    
    # Process each parameter
    for param in legacy_params:
        # Check if parameter exists in the file
        if param in content:
            # Pattern to match the parameter line
            pattern = rf"{param}.*?\n"
            # Remove the parameter line
            updated_content = re.sub(pattern, '', content)
            
            # If something was changed, update the content
            if updated_content != content:
                content = updated_content
                removed_params.append(param.rstrip(':'))
    
    # Write back the cleaned content
    with open(file_path, 'w') as f:
        f.write(content)
    
    return removed_params

def main():
    config_dir = 'config'
    config_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
    
    for file_name in config_files:
        file_path = os.path.join(config_dir, file_name)
        removed = cleanup_config_file(file_path)
        
        if removed:
            print(f"Removed from {file_name}: {', '.join(removed)}")
        else:
            print(f"No legacy parameters found in {file_name}")

if __name__ == '__main__':
    main() 