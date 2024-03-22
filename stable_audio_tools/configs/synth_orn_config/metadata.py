import json

# Assuming the JSON files are in the same directory as the provided path
# Adjust the base directory as necessary
# Assuming this function is defined in a module at the top level
def get_custom_metadata(info, audio):
    # Simplified logic that avoids complex operations or external state
    try:
        with open(f'/content/drive/MyDrive/colab_storage/audioWeather/Pied Currawong/' + info.get("relpath", "").split('/')[-1].replace('_P.wav', '.json'), 'r') as file:
            metadata_entry = json.load(file)
    except FileNotFoundError:
        return {
            "latitude": 0.0, "longitude": 0.0, "temperature": 0.0,
            "humidity": 0.0, "wind_speed": 0.0, "pressure": 0.0,
            "minutes_of_day": 0.0, "day_of_year": 0.0,
        }

    # Map the keys from the JSON structure to the required format
    metadata_for_entry = {
        "latitude": metadata_entry.get("coord", {}).get("lat", 0.0),
        "longitude": metadata_entry.get("coord", {}).get("lon", 0.0),
        "temperature": metadata_entry.get("main", {}).get("temp", 0.0),
        "humidity": metadata_entry.get("main", {}).get("humidity", 0.0),
        "wind_speed": metadata_entry.get("wind", {}).get("speed", 0.0),
        "pressure": metadata_entry.get("main", {}).get("pressure", 0.0),
        "minutes_of_day": metadata_entry.get("minutesOfDay", 0.0),
        "day_of_year": metadata_entry.get("dayOfYear", 0.0),
        
    }

    return metadata_for_entry