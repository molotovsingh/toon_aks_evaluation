import unittest
from pathlib import Path

from src.core.classification_prompt import (
    LABEL_DEFINITIONS,
    PromptPayload,
    build_classification_prompt,
)


class ClassificationPromptTests(unittest.TestCase):
    """Validate classification prompt assembly without hitting external APIs."""

    def test_build_classification_prompt_includes_expected_sections(self) -> None:
        sample_path = Path("test_documents/abc_xyz_contract_dispute.txt")
        document_text = sample_path.read_text().strip()

        payload = build_classification_prompt(
            document_title="ABC vs XYZ Contract Dispute Timeline",
            document_excerpt=document_text,
            prompt_version="test-v1",
        )

        self.assertIsInstance(payload, PromptPayload)
        self.assertEqual(payload.prompt_version, "test-v1")

        # System message must enforce JSON output schema.
        self.assertNotIn('{"classes": [<one or more labels>]}', payload.user_message)
        self.assertIn('{"classes": [<one or more labels>]', payload.system_message)
        self.assertIn("Respond with strict JSON", payload.system_message)

        # User message should enumerate labels and include few-shot examples.
        for definition in LABEL_DEFINITIONS:
            self.assertIn(definition, payload.user_message)
        self.assertIn("Example 1:", payload.user_message)
        self.assertIn("Example 2:", payload.user_message)
        self.assertIn("Document Title: ABC vs XYZ Contract Dispute Timeline", payload.user_message)
        self.assertIn(document_text[:60], payload.user_message)
        self.assertIn(document_text[-60:], payload.user_message)


if __name__ == "__main__":
    unittest.main()
