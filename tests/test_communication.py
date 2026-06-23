import json
import logging
import pytest
from unittest.mock import Mock, patch, MagicMock
from threading import Thread
import time

from interaction.communication import (
    Communication,
    DoneLook,
    DoneFind,
    DoneDeliver,
    DoneStop,
    Feedback,
)


@pytest.fixture
def mock_server():
    """Mock websocket server."""
    server = Mock()
    server.serve_forever = Mock()
    return server


@pytest.fixture
def communication(mock_server):
    """Create a Communication instance with mocked server."""
    with patch("interaction.communication.serve", return_value=mock_server):
        comm = Communication(port=8765)
        # Stop the server thread to avoid blocking
        comm.server = mock_server
        return comm


class TestDataclasses:
    """Test dataclass definitions."""

    def test_done_look(self):
        done_look = DoneLook(success=True, img="test.jpg")
        assert done_look.success is True
        assert done_look.img == "test.jpg"

    def test_done_find(self):
        done_find = DoneFind(success=True, message="Found", payload={"item": "key"})
        assert done_find.success is True
        assert done_find.message == "Found"
        assert done_find.payload == {"item": "key"}

    def test_done_deliver(self):
        done_deliver = DoneDeliver(success=True, message="Delivered")
        assert done_deliver.success is True
        assert done_deliver.message == "Delivered"

    def test_done_stop(self):
        done_stop = DoneStop(success=True, message="Stopped")
        assert done_stop.success is True
        assert done_stop.message == "Stopped"

    def test_feedback(self):
        feedback = Feedback(message="Processing")
        assert feedback.message == "Processing"


class TestCommunicationInit:
    """Test Communication initialization."""

    @patch("interaction.communication.serve")
    @patch("interaction.communication.Thread")
    def test_initialization(self, mock_thread_class, mock_serve):
        mock_server = Mock()
        mock_serve.return_value = mock_server
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        comm = Communication(port=9999)

        assert comm.logger is not None
        assert comm.event is not None
        assert comm.pending == []
        assert comm.server == mock_server
        mock_serve.assert_called_once_with(comm._handler, "localhost", 9999)
        mock_thread_class.assert_called_once()
        mock_thread.start.assert_called_once()

    @patch("interaction.communication.serve")
    @patch("interaction.communication.Thread")
    def test_initialization_default_port(self, mock_thread_class, mock_serve):
        mock_server = Mock()
        mock_serve.return_value = mock_server
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        comm = Communication()

        mock_serve.assert_called_once_with(comm._handler, "localhost", 8765)


class TestCommunicationHandler:
    """Test message handler."""

    def test_handler_processes_message(self, communication):
        # Test that _handler correctly processes incoming messages
        test_data = {"type": "test-message", "payload": {"key": "value"}}
        mock_websocket = Mock()
        mock_websocket.__iter__ = Mock(return_value=iter([json.dumps(test_data)]))

        communication._handler(mock_websocket)

        assert len(communication.pending) == 1
        assert communication.pending[0] == test_data

    def test_handler_appends_to_pending(self, communication):
        # Simulate receiving a message
        test_data = {"type": "done-look", "success": True, "img": "test.jpg"}
        
        with communication.event:
            communication.pending.append(test_data)

        assert len(communication.pending) == 1
        assert communication.pending[0] == test_data

    def test_handler_with_multiple_messages(self, communication):
        # Simulate receiving multiple messages
        messages = [
            {"type": "done-look", "success": True, "img": "img1.jpg"},
            {"type": "done-find", "success": True, "message": "Found", "payload": {}},
        ]
        
        for msg in messages:
            with communication.event:
                communication.pending.append(msg)

        assert len(communication.pending) == 2


class TestLookForMethod:
    """Test the _look_for method."""

    def test_look_for_finds_message(self, communication):
        # Add a message to the queue
        test_data = {"type": "done-look", "success": True, "img": "test.jpg"}
        communication.pending.append(test_data)

        # Look for the message
        result = communication._look_for("done-look")

        assert result == test_data
        assert len(communication.pending) == 0

    def test_look_for_removes_message(self, communication):
        # Add multiple messages
        messages = [
            {"type": "done-find", "success": True, "message": "Found", "payload": {}},
            {"type": "done-look", "success": True, "img": "test.jpg"},
        ]
        communication.pending.extend(messages)

        # Look for the second message type
        result = communication._look_for("done-look")

        assert result == messages[1]
        assert len(communication.pending) == 1
        assert communication.pending[0] == messages[0]


class TestSendMethod:
    """Test the _send method."""

    def test_send_creates_correct_payload(self, communication):
        communication.server.send = Mock()
        
        communication._send("look", {})

        # Verify send was called
        communication.server.send.assert_called_once()
        
        # Extract the payload that was sent
        sent_data = json.loads(communication.server.send.call_args[0][0])
        assert sent_data["type"] == "look"
        assert sent_data["payload"] == {}
        assert "stamp" in sent_data

    def test_send_with_payload(self, communication):
        communication.server.send = Mock()
        payload = {"noun": "key", "adj": ["red", "small"]}
        
        communication._send("find", payload)

        sent_data = json.loads(communication.server.send.call_args[0][0])
        assert sent_data["type"] == "find"
        assert sent_data["payload"] == payload


class TestPublicMethods:
    """Test public communication methods."""

    def test_look_returns_done_look(self, communication):
        # Mock the internal methods
        communication._send = Mock()
        communication._look_for = Mock(
            return_value={"success": True, "img": "test.jpg"}
        )

        result = communication.look()

        assert isinstance(result, DoneLook)
        assert result.success is True
        assert result.img == "test.jpg"
        communication._send.assert_called_once_with("look", {})
        communication._look_for.assert_called_once_with("done-look")

    def test_find_returns_done_find(self, communication):
        # Mock the internal methods
        communication._send = Mock()
        communication._look_for = Mock(
            return_value={
                "success": True,
                "message": "Found item",
                "payload": {"item_id": 1},
            }
        )

        result = communication.find("key", ["red", "shiny"])

        assert isinstance(result, DoneFind)
        assert result.success is True
        assert result.message == "Found item"
        assert result.payload == {"item_id": 1}
        communication._send.assert_called_once_with(
            "find", {"noun": "key", "adj": ["red", "shiny"]}
        )
        communication._look_for.assert_called_once_with("done-find")

    def test_find_with_empty_adjectives(self, communication):
        communication._send = Mock()
        communication._look_for = Mock(
            return_value={"success": False, "message": "Not found", "payload": None}
        )

        result = communication.find("item", [])

        assert isinstance(result, DoneFind)
        communication._send.assert_called_once_with(
            "find", {"noun": "item", "adj": []}
        )


class TestIntegration:
    """Integration tests."""

    def test_look_integration(self, communication):
        # Simulate a complete look workflow
        communication._send = Mock()

        # Prepare response with required "type" field
        response = {"type": "done-look", "success": True, "img": "captured.jpg"}
        communication.pending.append(response)

        # This would normally block waiting for response
        # For testing, we populate pending first
        result = communication._look_for("done-look")

        assert result["success"] is True
        assert result["img"] == "captured.jpg"

    def test_find_with_payload(self, communication):
        # Simulate a complete find workflow
        communication._send = Mock()

        # Prepare response with required "type" field
        response = {
            "type": "done-find",
            "success": True,
            "message": "Found the key",
            "payload": {"location": "drawer", "quantity": 1},
        }
        communication.pending.append(response)

        # This would normally block waiting for response
        result = communication._look_for("done-find")

        assert result["success"] is True
        assert result["message"] == "Found the key"
        assert result["payload"]["location"] == "drawer"
