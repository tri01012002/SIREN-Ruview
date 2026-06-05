#test_udp.py
import socket

print("🔍 Đang lắng nghe UDP port 5005... (Ctrl+C để dừng)")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5005))
sock.settimeout(5.0)

count = 0
try:
    while True:
        try:
            data, addr = sock.recvfrom(4096)
            count += 1
            print(f"📡 Nhận gói UDP từ {addr[0]} | Size: {len(data)} bytes | Tổng: {count}")
        except socket.timeout:
            print("⏳ Không nhận được gói nào trong 5 giây...")
except KeyboardInterrupt:
    print(f"\nĐã nhận tổng {count} gói tin.")