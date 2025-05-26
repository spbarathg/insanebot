"""
Mock classes for Solana components used in tests.
"""
class MockKeypair:
    def __init__(self, secret_key=None):
        self.secret_key = secret_key or bytes([0] * 32)
        self.public_key = MockPublicKey("mock_public_key")

    @classmethod
    def from_secret_key(cls, secret_key):
        return cls(secret_key)

class MockPublicKey:
    def __init__(self, address):
        self.address = address

    def __str__(self):
        return self.address

class MockTransaction:
    def __init__(self):
        self.instructions = []
        self.recent_blockhash = "mock_blockhash"
        
    def add(self, instruction):
        self.instructions.append(instruction)
        
    def sign(self, *signers):
        self.signatures = [signer.public_key for signer in signers]

class MockInstruction:
    def __init__(self, program_id, accounts, data):
        self.program_id = program_id
        self.accounts = accounts
        self.data = data

class MockAccountMeta:
    def __init__(self, pubkey, is_signer, is_writable):
        self.pubkey = pubkey
        self.is_signer = is_signer
        self.is_writable = is_writable 