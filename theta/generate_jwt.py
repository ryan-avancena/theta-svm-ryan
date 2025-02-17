import jwt
import time

# 🔹 Replace with your Apple Developer info
team_id = "NU7J7RLG4M"  # Your Team ID
key_id = "KS9VTAM9RV"   # Your Key ID
private_key = """
-----BEGIN PRIVATE KEY-----
MIGTAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBHkwdwIBAQQgaVPmKKZEHrU4OHRo
Jh907QckEGJAmEnSG3A+iV0J5J6gCgYIKoZIzj0DAQehRANCAATmrmGKJwDctHGG
NdReCQcA9aCOeypQfEqanx0b7mNr0tQJljWdW0A3HY8J4pMcBTxowrfGt7LcjgSg
ol9VqIpE
-----END PRIVATE KEY-----
"""

# Generate JWT token
jwt_payload = {
    "iss": team_id,
    "iat": int(time.time()),  # Current time in seconds
    "exp": int(time.time()) + 86400  # Token valid for 24 hours
}

headers = {
    "kid": key_id,  # Key ID
    "alg": "ES256"  # Algorithm
}

try:
    token = jwt.encode(jwt_payload, private_key, algorithm="ES256", headers=headers)
    print("🔑 Generated JWT Token:")
    print(token)
except Exception as e:
    print("❌ Error generating JWT token:")
    print(e)