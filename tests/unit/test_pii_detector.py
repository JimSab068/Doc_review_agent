from src.ingestion import ingest_text
from src.pii_detector import RegexPIIDetector
from src.schemas import DocumentType, PIIType


class TestRegexPIIDetector:
    def setup_method(self):
        self.detector = RegexPIIDetector()

    def test_detects_ssn(self):
        doc = ingest_text("Applicant SSN: 123-45-6789.", DocumentType.LOAN_APPLICATION, "a.txt")

        spans = self.detector.detect(doc)

        ssn_spans = [s for s in spans if s.field_type == PIIType.SSN]
        assert len(ssn_spans) == 1
        assert ssn_spans[0].raw_value == "123-45-6789"

    def test_detects_account_number(self):
        doc = ingest_text("Account #: 00123456789", DocumentType.BANK_STATEMENT, "a.txt")

        spans = self.detector.detect(doc)

        acct_spans = [s for s in spans if s.field_type == PIIType.ACCOUNT_NUMBER]
        assert len(acct_spans) == 1
        assert acct_spans[0].raw_value == "00123456789"

    def test_detects_date_of_birth(self):
        doc = ingest_text("DOB: 04/12/1990", DocumentType.LOAN_APPLICATION, "a.txt")

        spans = self.detector.detect(doc)

        dob_spans = [s for s in spans if s.field_type == PIIType.DATE_OF_BIRTH]
        assert len(dob_spans) == 1
        assert dob_spans[0].raw_value == "04/12/1990"

    def test_detects_phone_number(self):
        doc = ingest_text("Contact: (555) 123-4567", DocumentType.LOAN_APPLICATION, "a.txt")

        spans = self.detector.detect(doc)

        phone_spans = [s for s in spans if s.field_type == PIIType.PHONE]
        assert len(phone_spans) == 1
        assert phone_spans[0].raw_value == "(555) 123-4567"

    def test_phone_regex_keeps_opening_parenthesis(self):
        """Regression test: a leading \\b in the old pattern couldn't form
        a boundary between a space and '(', so '(201) 555-0134' matched as
        '201) 555-0134' -- dropping the opening paren. Caught via a live
        Gemini run, not by the original unit tests, which used phone
        numbers at the very start of a string."""
        text = "Phone: (201) 555-0134"
        doc = ingest_text(text, DocumentType.LOAN_APPLICATION, "a.txt")

        spans = self.detector.detect(doc)

        phone_spans = [s for s in spans if s.field_type == PIIType.PHONE]
        assert len(phone_spans) == 1
        assert phone_spans[0].raw_value == "(201) 555-0134"

    def test_detects_email(self):
        doc = ingest_text("Email: jane.doe@example.com", DocumentType.LOAN_APPLICATION, "a.txt")

        spans = self.detector.detect(doc)

        email_spans = [s for s in spans if s.field_type == PIIType.EMAIL]
        assert len(email_spans) == 1
        assert email_spans[0].raw_value == "jane.doe@example.com"

    def test_detects_multiple_pii_types_in_one_document(self):
        text = "SSN: 987-65-4321, DOB: 01/01/1985, Email: a@b.com"
        doc = ingest_text(text, DocumentType.LOAN_APPLICATION, "a.txt")

        spans = self.detector.detect(doc)
        found_types = {s.field_type for s in spans}

        assert PIIType.SSN in found_types
        assert PIIType.DATE_OF_BIRTH in found_types
        assert PIIType.EMAIL in found_types

    def test_span_offsets_are_correct(self):
        text = "prefix 123-45-6789 suffix"
        doc = ingest_text(text, DocumentType.LOAN_APPLICATION, "a.txt")

        spans = self.detector.detect(doc)
        ssn_span = next(s for s in spans if s.field_type == PIIType.SSN)

        assert text[ssn_span.start_char : ssn_span.end_char] == "123-45-6789"

    def test_no_false_positive_on_clean_text(self):
        doc = ingest_text(
            "The applicant requested a loan for home improvement purposes.",
            DocumentType.LOAN_APPLICATION,
            "a.txt",
        )

        spans = self.detector.detect(doc)

        assert spans == []

    def test_detects_pii_across_multiple_pages(self):
        from src.schemas import Document, Page

        doc = Document(
            doc_id="multi-page",
            doc_type=DocumentType.BANK_STATEMENT,
            source_filename="a.txt",
            pages=[
                Page(page_number=1, text="SSN: 111-22-3333"),
                Page(page_number=2, text="Email: x@y.com"),
            ],
        )

        spans = self.detector.detect(doc)

        assert any(s.page_number == 1 and s.field_type == PIIType.SSN for s in spans)
        assert any(s.page_number == 2 and s.field_type == PIIType.EMAIL for s in spans)

    def test_uses_injected_ner_detector_when_provided(self):
        class FakeNER:
            def detect(self, text):
                return [(PIIType.PERSON_NAME, "Jane Doe", 0, 8)]

        detector = RegexPIIDetector(ner_detector=FakeNER())
        doc = ingest_text("Jane Doe applied for a loan.", DocumentType.LOAN_APPLICATION, "a.txt")

        spans = detector.detect(doc)

        name_spans = [s for s in spans if s.field_type == PIIType.PERSON_NAME]
        assert len(name_spans) == 1
        assert name_spans[0].detector == "ner"

    def test_grouped_regex_offset_slicing(self):
        """Verifies start_char and end_char accurately slice values for patterns with capture groups (e.g., Account Number and DOB)."""
        text = "Primary Acct #: 9876543210 and DOB: 05/20/1988 in system."
        doc = ingest_text(text, DocumentType.LOAN_APPLICATION, "a.txt")

        spans = self.detector.detect(doc)

        acct_span = next(s for s in spans if s.field_type == PIIType.ACCOUNT_NUMBER)
        dob_span = next(s for s in spans if s.field_type == PIIType.DATE_OF_BIRTH)

        # Slice text using detected character bounds
        assert text[acct_span.start_char : acct_span.end_char] == "9876543210"
        assert text[dob_span.start_char : dob_span.end_char] == "05/20/1988"

    def test_pii_span_metadata_attributes(self):
        """Verifies confidence, detector name, doc_id, and page_number attributes on generated PIISpan objects."""
        doc = ingest_text("SSN: 000-11-2222", DocumentType.LOAN_APPLICATION, "a.txt", doc_id="meta-doc-123")

        spans = self.detector.detect(doc)
        ssn_span = spans[0]

        assert ssn_span.doc_id == "meta-doc-123"
        assert ssn_span.page_number == 1
        assert ssn_span.confidence == 0.95
        assert ssn_span.detector == "regex:ssn"

    def test_regex_pattern_variations(self):
        """Verifies pattern matching across common formatting variations."""
        text = "Acct. 1234567890, Date of Birth 1/2/95, Phone +1-800-555-0199 and 555.123.4567"
        doc = ingest_text(text, DocumentType.LOAN_APPLICATION, "a.txt")

        spans = self.detector.detect(doc)
        found_types = [s.field_type for s in spans]

        assert PIIType.ACCOUNT_NUMBER in found_types
        assert PIIType.DATE_OF_BIRTH in found_types
        assert found_types.count(PIIType.PHONE) == 2

    def test_detects_multiple_instances_of_same_pii_type(self):
        """Verifies multiple instances of the same PII type on a single page are all detected."""
        text = "Primary email: alice@example.com, Secondary email: bob@example.com"
        doc = ingest_text(text, DocumentType.LOAN_APPLICATION, "a.txt")

        spans = self.detector.detect(doc)
        email_spans = [s for s in spans if s.field_type == PIIType.EMAIL]

        assert len(email_spans) == 2
        assert {s.raw_value for s in email_spans} == {"alice@example.com", "bob@example.com"}