import os
# ray_address = os.environ['RAY_ADDRESS']
# ray_address = f"ray://{os.environ['RAY_ADDRESS_HOST']}:6380"
if os.environ['RAY_MULTIPLE_DEPLOYMENT'] =="TRUE":
    ray_address = f"{os.environ['RAY_ADDRESS_HOST']}"
    port = ray_address.split(":")[1]
    os.environ['RAY_ADDRESS'] = ray_address
    dashboard_port = f"{os.environ['RAY_DASHBOARD_PORT']}"
    ray_client_server_port = f"{os.environ['RAY_CLIENT_SERVER_PORT']}"
    dashboard_grpc_port = f"{os.environ['DASHBOARD_GRPC_PORT']}"

    dashboard_agent_listen_port = f"{os.environ['DASHBOARD_AGENT_LISTEN_PORT']}"
    node_manager_port= f"{os.environ['NODE_MANAGER_PORT']}"
    object_manager_port = f"{os.environ['OBJECT_MANAGER_PORT']}"

        # - RAY_RAY_CLIENT_SERVER_PORT=30795 #10001
        # - OBJECT_MANAGER_PORT = 31844# 8076
        # - NODE_MANAGER_PORT = 31354 # 8077
        # - DASHBOARD_GRPC_PORT= 31724#8078
        # - DASHBOARD_AGENT_LISTEN_PORT= 30015# 8079


    print(f"using ray address: {ray_address }, Port:{port}, dashboard:{dashboard_port}")
    start_cmd = f"ray start --address={ray_address} --port={port} --dashboard-port={dashboard_port} --ray-client-server-port={ray_client_server_port} --dashboard-agent-grpc-port={dashboard_grpc_port} --object-manager-port={object_manager_port} --node-manager-port={object_manager_port} --dashboard-agent-listen-port={dashboard_agent_listen_port} --min-worker-port=13105 --max-worker-port=14305 --block"
    print(start_cmd)
    os.system(start_cmd)
else:
    ray_address = f"{os.environ['RAY_ADDRESS_HOST']}:6380"
    os.environ['RAY_ADDRESS'] = ray_address
    
    # ray-client-server-port
    # --dashboard-grpc-port
    

    print(f"using ray address: {ray_address }")
    os.system(f"ray start --address={ray_address} --object-manager-port=8076 --node-manager-port=8077 --dashboard-agent-grpc-port=8078 --dashboard-agent-listen-port=8079 --min-worker-port=13105 --max-worker-port=14305 --block")

# os.system(f"ray start --address={ray_address} --object-manager-port=8076 --node-manager-port=8077 --dashboard-agent-grpc-port=8078 --dashboard-agent-listen-port=8079 --min-worker-port=10002 --max-worker-port=10007 --block")
# os.system(f"ray start --address={ray_address} --object-manager-port=8076 --node-manager-port=8077 --dashboard-agent-grpc-port=8078 --dashboard-agent-listen-port=8079 --min-worker-port=13105 --max-worker-port=14305 --block")
