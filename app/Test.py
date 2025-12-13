from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_upload_valid_file():

    with open("samples/sample.odt", "wb") as f:
        f.write(b"Dummy ODT content")

    with open("samples/sample.odt", "rb") as f:
        response = client.post(
            "/upload",
            files={"file": ("sample.odt", f, "application/vnd.oasis.opendocument.text")}
        )

    assert response.status_code == 200
    assert "filename" in response.json()


def test_upload_invalid_file():

    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"invalid", "text/plain")}
    )

    assert response.status_code == 400


def test_system_health():

    response = client.get("/")
    assert response.status_code == 200


def test_full_module_accuracy():

    tests = [
        test_upload_valid_file,
        test_upload_invalid_file,
        test_system_health,
    ]

    passed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError:
            pass

    accuracy = (passed / len(tests)) * 100

    print(f"Final module accuracy: {accuracy}%")

    assert accuracy >= 90
