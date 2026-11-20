"""校准专用 ROBOTIS SDK 适配器；不提供启用扭矩或运动目标接口。"""

import time

from dynamixel_sdk.robotis_def import COMM_RX_CORRUPT, COMM_RX_TIMEOUT, INST_PING, INST_READ, INST_WRITE
from dynamixel_sdk.protocol2_packet_handler import Protocol2PacketHandler, PKT_INSTRUCTION, PKT_PARAMETER0, ERRNUM_CRC


REGISTERS = {
    "Model_Number": (0, 2), "Drive_Mode": (10, 1), "Operating_Mode": (11, 1),
    "Homing_Offset": (20, 4), "Max_Position_Limit": (48, 4),
    "Min_Position_Limit": (52, 4), "Torque_Enable": (64, 1),
    "Hardware_Error_Status": (70, 1), "Present_Position": (132, 4),
    "Present_Input_Voltage": (144, 2), "Present_Temperature": (146, 1),
}
WRITABLE = {
    "Drive_Mode": (0, 1), "Operating_Mode": (4, 4), "Torque_Enable": (0, 0),
    "Homing_Offset": (-1_044_479, 1_044_479),
    "Min_Position_Limit": (0, 4095), "Max_Position_Limit": (0, 4095),
}


class DynamixelCommunicationError(RuntimeError):
    def __init__(self, result, message):
        super().__init__(message)
        self.result = result


class ValidatedProtocol2PacketHandler(Protocol2PacketHandler):
    def txRxPacket(self, port, txpacket):
        try:
            packet, result, error = super().txRxPacket(port, txpacket)
        except IndexError:
            port.is_using = False
            return None, COMM_RX_CORRUPT, 0
        if result == 0 and error == 0:
            instruction = txpacket[PKT_INSTRUCTION]
            if instruction == INST_READ:
                size = int.from_bytes(bytes(txpacket[PKT_PARAMETER0 + 2:PKT_PARAMETER0 + 4]), 'little')
            elif instruction in (INST_PING, INST_WRITE):
                size = 3 if instruction == INST_PING else 0
            else:
                return packet, result, error
            # SDK 已验 CRC、ID 并去除填充；CRC 两字节不属于寄存器数据。
            # 迟到的写入 ACK 只有 11 字节，不能当作同 ID 的 READ 应答。
            if packet is None or len(packet) != 11 + size:
                result = COMM_RX_CORRUPT
        return packet, result, error


class KochCalibrationBus:
    def __init__(self, port, motors):
        self.port, self.motors = port, dict(motors)
        self.is_connected = False
        self._port = self._packet = None

    def _check(self, comm, error, operation):
        if comm != 0:
            raise DynamixelCommunicationError(comm, f"{self.port} {operation}: {self._packet.getTxRxResult(comm)}")
        if error:
            raise RuntimeError(f"{self.port} {operation}: {self._packet.getRxPacketError(error)}")

    def connect(self):
        from dynamixel_sdk import PortHandler

        if self.is_connected:
            raise RuntimeError("串口已经连接")
        self._port, self._packet = PortHandler(self.port), ValidatedProtocol2PacketHandler()
        try:
            if not self._port.openPort() or not self._port.setBaudRate(1_000_000):
                raise RuntimeError(f"无法打开 {self.port} 或设置 1Mbps")
            for name, (motor_id, expected) in self.motors.items():
                for attempt in range(3):
                    model, comm, error = self._packet.ping(self._port, motor_id)
                    if not (comm in (COMM_RX_TIMEOUT, COMM_RX_CORRUPT) or
                            (comm == 0 and error == ERRNUM_CRC)) or attempt == 2:
                        break
                    self._port.clearPort()
                    time.sleep(0.01)
                self._check(comm, error, f"PING ID {motor_id}")
                if model != expected:
                    raise RuntimeError(f"{name} ID {motor_id} 型号 {model}，预期 {expected}")
            self.is_connected = True
        except BaseException:
            self.disconnect(disable_torque=False)
            raise

    def disconnect(self, disable_torque=False):
        try:
            if disable_torque and self.is_connected:
                self.disable_torque()
        finally:
            try:
                # 打开失败时 SDK 的 ser 为 None，closePort() 会掩盖原始串口异常。
                if self._port is not None and self._port.ser is not None:
                    self._port.closePort()
            finally:
                self._port = None
                self.is_connected = False

    def _require_connection(self, normalize):
        if normalize:
            raise ValueError("校准接口仅支持原始寄存器数据")
        if not self.is_connected:
            raise RuntimeError("串口未连接")

    def read(self, register, name, normalize=False):
        self._require_connection(normalize)
        address, width = REGISTERS[register]
        value = self._read_register(register, name, address, width)
        if register in ("Homing_Offset", "Present_Position") and value >= 2**31:
            value -= 2**32
        return value

    def _read_register(self, register, name, address, width):
        operation = f"读取 ID {self.motors[name][0]} {name}.{register}"
        for attempt in range(3):
            try:
                data, comm, error = self._packet.readTxRx(self._port, self.motors[name][0], address, width)
            except IndexError:
                self._port.is_using = False
                reason = "SDK 收到不完整状态包"
            else:
                if comm in (COMM_RX_TIMEOUT, COMM_RX_CORRUPT):
                    reason = self._packet.getTxRxResult(comm)
                elif comm == 0 and error == ERRNUM_CRC:
                    # 电机明确报告指令 CRC 错误；仅读操作可以有限重试。
                    reason = self._packet.getRxPacketError(error)
                else:
                    # 必须先检查故障码和长度，再解码；SDK 定长读取会先索引缺失数据。
                    self._check(comm, error, operation)
                    if len(data) == width:
                        return int.from_bytes(bytes(data), 'little')
                    reason = f"状态数据长度 {len(data)}，预期 {width} 字节"
            if attempt < 2:
                self._port.is_using = False
                self._port.clearPort()
                time.sleep(0.01)
        raise RuntimeError(f"{self.port} {operation}: 连续 3 次读取失败；{reason}")

    def write(self, register, name, value, normalize=False):
        self._require_connection(normalize)
        if register not in WRITABLE:
            raise ValueError(f"校准禁止写入 {register}")
        low, high = WRITABLE[register]
        if not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{register} 值必须在 {low}..{high} 内")
        address, width = REGISTERS[register]
        self._write_register(register, name, address, width, value)

    def _write_register(self, register, name, address, width, value):
        encoded = value & ((1 << (width * 8)) - 1)
        method = getattr(self._packet, f"write{width}ByteTxRx")
        comm, error = method(self._port, self.motors[name][0], address, encoded)
        operation = f"写入 ID {self.motors[name][0]} {name}.{register}"
        if comm not in (COMM_RX_TIMEOUT, COMM_RX_CORRUPT):
            self._check(comm, error, operation)
            return
        # 只写一次：丢失 ACK 不代表未执行。回读未确认时停止，不补发运动命令。
        try:
            actual = self._read_register(register, name, address, width)
        except Exception as exc:
            raise DynamixelCommunicationError(comm, f'{self.port} {operation}: 应答丢失且无法回读确认；{exc}') from exc
        if actual != encoded:
            raise DynamixelCommunicationError(comm, f'{self.port} {operation}: 应答丢失，回读 {actual}，'
                                              f'预期 {encoded}；未自动重发')
        fault = self._read_register('Hardware_Error_Status', name, 70, 1)
        if fault:
            raise RuntimeError(f'{self.port} {operation}: 回读值匹配，但电机故障码为 {fault}')

    def sync_read(self, register, normalize=False):
        # 手动校准逐个读取即可；保留向导接口名，不声称同步采样。
        return {name: self.read(register, name, normalize) for name in self.motors}

    def disable_torque(self, num_retry=0):
        failures = []
        for name in self.motors:
            for attempt in range(num_retry + 1):
                try:
                    self.write("Torque_Enable", name, 0)
                    break
                except Exception as exc:
                    if attempt == num_retry:
                        failures.append(str(exc))
        if failures:
            raise RuntimeError("关闭扭矩失败：" + "; ".join(failures))

    def set_operating_mode(self, mode):
        if mode != "extended_position":
            raise ValueError(mode)
        for name in self.motors:
            self.write("Operating_Mode", name, 4)

    def reset_calibration(self):
        for name in self.motors:
            for register, value in (("Homing_Offset", 0), ("Min_Position_Limit", 0),
                                    ("Max_Position_Limit", 4095)):
                self.write(register, name, value)

    def write_calibration(self, calibration):
        for name, cal in calibration.items():
            self.write("Homing_Offset", name, cal.homing_offset)
            self.write("Min_Position_Limit", name, cal.range_min)
            self.write("Max_Position_Limit", name, cal.range_max)

    def read_calibration(self):
        from lerobot.motors import MotorCalibration

        return {name: MotorCalibration(
            id=motor_id, drive_mode=self.read("Drive_Mode", name),
            homing_offset=self.read("Homing_Offset", name),
            range_min=self.read("Min_Position_Limit", name),
            range_max=self.read("Max_Position_Limit", name),
        ) for name, (motor_id, _) in self.motors.items()}

    def record_ranges_of_motion(self, names):
        import msvcrt

        mins, maxes = {}, {}
        while True:
            for name in names:
                position = self.read("Present_Position", name)
                if abs(position) > 8191:
                    raise RuntimeError(f"{name} 位置异常：{position}")
                mins[name] = min(mins.get(name, position), position)
                maxes[name] = max(maxes.get(name, position), position)
                if maxes[name] - mins[name] > 4095:
                    raise RuntimeError(f"{name} 采集跨度超过一圈，请检查读数或重新校准")
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key == "\x03":
                    raise KeyboardInterrupt
                if key in ("\r", "\n"):
                    return mins, maxes
            time.sleep(0.02)
