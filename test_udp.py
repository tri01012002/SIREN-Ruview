import socket
import time
import struct
from datetime import datetime

# Cấu hình
UDP_IP = "0.0.0.0"      # Nghe tất cả interface
UDP_PORT = 5005         # Port mặc định của ESP32 CSI trong project

print(f"🚀 Listening for CSI UDP packets on {UDP_IP}:{UDP_PORT}")
print("Đợi ESP32 gửi data... (nhấn Ctrl+C để dừng)")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(5.0)   # timeout mỗi lần recv

packet_count = 0
start_time = time.time()

try:
    while True:
        try:
            data, addr = sock.recvfrom(4096)  # Buffer đủ lớn cho CSI frame
            packet_count += 1
            now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            print(f"[{now}] ✅ Received CSI packet from {addr[0]}:{addr[1]} | Size: {len(data)} bytes")
            
            # In vài byte đầu để debug (thường bắt đầu bằng magic/header)
            if len(data) >= 16:
                header = data[:16]
                print(f"   Header: {header.hex()[:32]}...")
            
            # Thống kê tốc độ
            if packet_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = packet_count / elapsed
                print(f"   📊 Total: {packet_count} packets | Rate: {rate:.1f} pkt/s")
                
        except socket.timeout:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ No packet received in last 5s...")
            continue
            
except KeyboardInterrupt:
    print("\n\n🛑 Stopped by user")
    elapsed = time.time() - start_time
    print(f"Thống kê: {packet_count} packets in {elapsed:.1f}s → {packet_count/elapsed:.1f} packets/s")
finally:
    sock.close()