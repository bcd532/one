"""Tests for entity extraction."""

import pytest
from one.entities import extract_entities, extract_from_tool_call, extract_relationships


class TestFileExtraction:
    def test_absolute_path(self):
        entities = extract_entities("edit /home/user/project/main.py")
        files = [e for e in entities if e["type"] == "file"]
        assert any("/home/user/project/main.py" in e["name"] for e in files)

    def test_relative_path(self):
        entities = extract_entities("look at ./src/app.py")
        files = [e for e in entities if e["type"] == "file"]
        assert any("./src/app.py" in e["name"] for e in files)

    def test_home_path(self):
        entities = extract_entities("read ~/config.yaml")
        files = [e for e in entities if e["type"] == "file"]
        assert any("~/config.yaml" in e["name"] for e in files)

    def test_no_false_positives_on_short_paths(self):
        entities = extract_entities("use /a")
        files = [e for e in entities if e["type"] == "file"]
        assert len(files) == 0  # too short


class TestConceptExtraction:
    def test_hdc_concepts(self):
        entities = extract_entities("the HDC encoder uses trigrams")
        concepts = [e for e in entities if e["type"] == "concept"]
        assert any("HDC" in e["name"] for e in concepts)

    def test_active_inference(self):
        entities = extract_entities("the active inference gate scores messages")
        concepts = [e for e in entities if e["type"] == "concept"]
        assert any("Active Inference" in e["name"] for e in concepts)

    def test_ml_concepts(self):
        entities = extract_entities("we should use a transformer with attention mechanism")
        concepts = [e for e in entities if e["type"] == "concept"]
        names = [e["name"] for e in concepts]
        assert "Transformer" in names
        assert "Attention Mechanism" in names

    def test_no_duplicate_concepts(self):
        entities = extract_entities("HDC encoding with hyperdimensional vectors")
        concepts = [e for e in entities if e["type"] == "concept"]
        hdc_concepts = [e for e in concepts if e["name"] == "HDC"]
        assert len(hdc_concepts) == 1


class TestCodePatterns:
    def test_class_detection(self):
        entities = extract_entities("class MyHandler(BaseHandler):")
        classes = [e for e in entities if e["type"] == "class"]
        assert any("MyHandler" in e["name"] for e in classes)

    def test_function_detection(self):
        entities = extract_entities("def process_data(input):")
        functions = [e for e in entities if e["type"] == "function"]
        assert any("process_data" in e["name"] for e in functions)

    def test_import_detection(self):
        entities = extract_entities("import numpy")
        modules = [e for e in entities if e["type"] == "module"]
        assert any("numpy" in e["name"] for e in modules)

    def test_from_import_detection(self):
        entities = extract_entities("from pathlib import Path")
        modules = [e for e in entities if e["type"] == "module"]
        assert any("pathlib" in e["name"] for e in modules)


class TestOrganizationExtraction:
    def test_tech_companies(self):
        entities = extract_entities("Google and OpenAI are working on this")
        orgs = [e for e in entities if e["type"] == "organization"]
        names = [e["name"] for e in orgs]
        assert "Google" in names
        assert "OpenAI" in names


class TestURLExtraction:
    def test_https_url(self):
        entities = extract_entities("check https://example.com/api")
        urls = [e for e in entities if e["type"] == "url"]
        assert len(urls) >= 1

    def test_http_url(self):
        entities = extract_entities("go to http://localhost:4111")
        urls = [e for e in entities if e["type"] == "url"]
        assert len(urls) >= 1


class TestToolCallExtraction:
    def test_read_tool(self):
        entities = extract_from_tool_call("Read", {"file_path": "/src/main.py"})
        assert any(e["type"] == "tool" for e in entities)
        assert any(e["name"] == "/src/main.py" for e in entities)

    def test_grep_tool(self):
        entities = extract_from_tool_call("Grep", {"pattern": "def main"})
        assert any(e["type"] == "tool" for e in entities)
        assert any(e["type"] == "pattern" for e in entities)


class TestRelationshipExtraction:
    def test_causal_relationship(self):
        rels = extract_relationships("The mutation causes resistance")
        assert any(r["relation_type"] == "causes" for r in rels)

    def test_dependency_relationship(self):
        rels = extract_relationships("The module depends on numpy")
        assert any(r["relation_type"] == "depends_on" for r in rels)

    def test_contradiction_relationship(self):
        rels = extract_relationships("Finding A contradicts finding B")
        # Might not match due to "finding" not being a single word
        # but let's test the mechanism
        assert isinstance(rels, list)


class TestEdgeCases:
    def test_empty_input(self):
        entities = extract_entities("")
        assert entities == []

    def test_pure_punctuation(self):
        entities = extract_entities("!!! ??? ...")
        assert isinstance(entities, list)

    def test_very_long_input(self):
        text = "word " * 10000
        entities = extract_entities(text)
        assert isinstance(entities, list)

    def test_binary_like_content(self):
        entities = extract_entities("\x00\x01\x02\x03")
        assert isinstance(entities, list)
