


# 从文件中读取 IP 地址
SERVER_IP=10.49.134.111
# 验证 IP 地址格式（可选）
if [[ ! $SERVER_IP =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Error: Invalid IP address format in $IP_FILE"
  exit 1
fi

# 目标端口和本地监听端口
TARGET_PORT=8501
LOCAL_PORT=9120

# 打印信息
echo "Setting up socat to forward local port $LOCAL_PORT to $SERVER_IP:$TARGET_PORT"

# 启动 socat
socat TCP-LISTEN:$LOCAL_PORT,fork TCP:$SERVER_IP:$TARGET_PORT 
# SOCAT_PID=$!

# 打印成功信息
# echo "Socat is running with PID $SOCAT_PID. Forwarding $LOCAL_PORT to $SERVER_IP:$TARGET_PORT"

# 可选：等待用户手动终止
# read -p "Press Enter to stop socat and exit..."

# 可选：停止 socat
# kill $SOCAT_PID
