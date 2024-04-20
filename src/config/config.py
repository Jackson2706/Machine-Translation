import json

def get_data_from_json(json_file):
    """
    Read data from a JSON file.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        dict: Dictionary containing the data read from the JSON file.
    """
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    return data

# Ví dụ sử dụng
if __name__ == "__main__":
    json_file = "config.json"  # Đường dẫn tới tập tin JSON

    data = get_data_from_json(json_file)
    print(data)
