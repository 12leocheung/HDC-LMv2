"""
ESP_LLM  --  talk to the HDC language model running on your ESP32.

Plug the board in over USB, run this, and type prompts.
    ESP_LLM.exe          (finds the board by itself)
    ESP_LLM.exe COM5     (use a specific port)
"""
import sys
import time
import traceback

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("pyserial is missing. Run:  python -m pip install pyserial")
    input("Press Enter to close...")
    sys.exit(1)

BAUD = 115200
# USB chips that ESP32 boards commonly use
KNOWN_CHIPS = {0x10C4: "CP210x", 0x1A86: "CH340", 0x0403: "FTDI", 0x303A: "Espressif USB"}


def pick_port():
    if len(sys.argv) > 1:
        return sys.argv[1]
    ports = list(list_ports.comports())
    likely = [p for p in ports if p.vid in KNOWN_CHIPS]
    choices = likely or ports
    if not choices:
        return None
    if len(choices) == 1:
        return choices[0].device
    print("Several serial ports found:")
    for i, p in enumerate(choices):
        print(f"  {i + 1}) {p.device}  {p.description}")
    while True:
        answer = input("Which one is the ESP32? (number) ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1].device


def wait_for(ser, prefixes, timeout):
    """Read lines until one starts with one of the prefixes. Returns it, or None."""
    end = time.time() + timeout
    while time.time() < end:
        raw = ser.readline()
        if not raw:
            continue
        text = raw.decode("utf-8", "replace").strip()
        for p in prefixes:
            if text.startswith(p):
                return text
    return None


def reset_board(ser):
    """Press the board's reset button electronically (works on most dev boards)."""
    try:
        ser.dtr = False
        ser.rts = True
        time.sleep(0.15)
        ser.rts = False
    except Exception:
        pass


def connect(port):
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = BAUD
    ser.timeout = 0.3
    try:
        ser.dtr = False
        ser.rts = False
        ser.open()
    except serial.SerialException as e:
        print(f"Could not open {port}: {e}")
        print("Close Thonny (or any other program using the board) and try again.")
        return None

    ser.reset_input_buffer()
    # Is the model already running? Ask it.
    ser.write(b"/ping\n")
    if wait_for(ser, ["HDC:"], 1.5):
        return ser

    # Not answering: restart the board so it runs main.py, then wait for it to load.
    print("Starting the model on the board...")
    reset_board(ser)
    if wait_for(ser, ["HDC_READY"], 25):
        return ser
    print("The board did not start the model. Check that:")
    print("  - main.py and the model files are on the board")
    print("  - Thonny is closed")
    print("  - if nothing else works, press the board's RST/EN button once and run this again")
    ser.close()
    return None


def main():
    print("ESP_LLM - HDC language model on ESP32")
    port = pick_port()
    if not port:
        print("No serial ports found. Plug the board in (check the USB cable carries data).")
        return
    print(f"Connecting to {port} ...")
    ser = connect(port)
    if not ser:
        return
    print("Ready. Type a prompt and press Enter. Type 'exit' to quit.\n")
    while True:
        start = time.time()
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.lower() in ("exit", "quit"):
            break
        if not prompt:
            continue
        try:
            ser.reset_input_buffer()
            ser.write((prompt + "\n").encode("utf-8"))
            reply = wait_for(ser, ["HDC:", "HDC_ERR:"], 60)
        except (serial.SerialException, OSError) as e:
            print(f"Lost the connection to the board ({e}). Reconnecting...")
            try:
                ser.close()
            except Exception:
                pass
            ser = connect(port)
            if not ser:
                print("Could not reconnect. Check the cable and run this again.")
                break
            print("Reconnected. Type your prompt again.")
            continue
        except Exception:
            traceback.print_exc()
            continue

        if reply is None:
            print("(no answer from the board)")
        elif reply.startswith("HDC_ERR:"):
            print("Board error: " + reply[8:])
        else:
            text = reply[4:].strip()
            print(prompt + " " + text + f" (took {time.time() - start:.2f}s)" if text else "(no prediction for that)" + f" (took {time.time() - start:.2f}s)")
            print("\n")
    ser.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nSomething went wrong:")
        traceback.print_exc()
    input("\nPress Enter to close...")