import requests

# Use the latest generated JWT token
server_token = "eyJraWQiOiJRUDcyTUFKUU5KIiwidHlwIjoiSldUIiwiYWxnIjoiRVMyNTYifQ.eyJpc3MiOiJOVTdKN1JMRzRNIiwiaWF0IjoxNzM5NDA3MjA1LCJleHAiOjE3NDAwMzgzOTl9.KEP_eHllmzc_yyzEbc9DFuw3tMOeYSRb4TRgbxELIDr_tjqd4L3U3kkMeRPgL42r-qifpBRsmSMkKGZ1_DywuA"

# Address to geocode
address = "1600 Amphitheatre Parkway, Mountain View, CA"
url = f"https://maps-api.apple.com/v1/geocode?q={address}"

# Set up request headers
headers = {
    "Authorization": f"Bearer {server_token}"
}

# Send GET request
response = requests.get(url, headers=headers)

# Print response
print(response.json())  # Expected output: latitude & longitude data
