from web3 import Web3
from eth_utils import to_bytes
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Web3 with HTTP Provider
alchemy_url = os.getenv('ALCHEMY_URL')

logging.basicConfig(level=logging.INFO)
logging.info("Connecting to Ethereum node at: %s", alchemy_url)

if not alchemy_url:
    raise EnvironmentError("ALCHEMY_URL environment variable not set.")

w3 = Web3(Web3.HTTPProvider(alchemy_url))

if not w3.is_connected():
    raise ConnectionError("Failed to connect to the Ethereum node.")

def decode_revert_reason(revert_code, contract_address):
    try:
        # Validate input
        if not hasattr(revert_code, 'message') or not isinstance(revert_code.message, str):
            raise ValueError("Invalid revert_code format. Expected a message attribute.")

        # Convert revert reason to bytes
        revert_bytes = to_bytes(hexstr=revert_code.message)
        logging.info("Converted revert code to bytes.")

        # Simulate contract call to extract revert reason
        revert_msg = w3.eth.call({
            'to': contract_address,
            'data': revert_bytes
        }, 'latest')  # Ensure to specify the block context
        logging.info("Simulated contract call. Revert message: %s", revert_msg)

        # Decode revert message using ABI
        decoded_message = w3.codec.decode_single('string', revert_msg)
        logging.info("Revert message decoded successfully: %s", decoded_message)
        
        # If decoded_message is a tuple, extract the first element
        if isinstance(decoded_message, tuple):
            decoded_message = decoded_message[0]
        
        return decoded_message

    except Exception as e:
        logging.error("Failed to decode revert reason: %s", str(e))
        return None

class SomeCustomErrorObject:
    def __init__(self, message):
        self.message = message

# Example usage
revert_code = SomeCustomErrorObject(message="0x6c661d58")
contract_address = os.getenv('SEPOLIA_URL')

decoded_message = decode_revert_reason(revert_code, contract_address)
if decoded_message:
    print("Revert reason:", decoded_message)
else:
    print("Failed to decode the revert reason.")
