"""Sensitivity-routed embedding layer (Milestone A).

Public path  -> Voyage (voyage-context-3), contextual, document-atomic (R2).
Private path -> local bge-m3-GGUF via llama-cpp-python.
Routed per document by a `private` flag; see scribbleslm.routing.
"""
