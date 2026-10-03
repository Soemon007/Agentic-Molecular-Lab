PROPOSALS = {"type": "object", "required": ["proposals"], "properties": {"proposals": {
    "type": "array", "items": {"type": "object", "required": ["parent_id", "smiles"], "properties": {
        "parent_id": {"type": "string", "description": "id of the beam member this molecule is derived from"},
        "smiles": {"type": "string"}}}}}}
