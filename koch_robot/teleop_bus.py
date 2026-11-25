"""遥操作专用运动接口；校准总线的写入限制保持不变。"""
import time

from dynamixel_sdk.robotis_def import COMM_SUCCESS, COMM_RX_CORRUPT, COMM_RX_TIMEOUT
from dynamixel_sdk.protocol2_packet_handler import ERRNUM_CRC, PKT_ID, PKT_ERROR, PKT_PARAMETER0

from koch_robot.calibration_bus import DynamixelCommunicationError, KochCalibrationBus


MOTION_REGISTERS = {
    'Profile_Acceleration': (108, 4, 0, 32767),
    'Profile_Velocity': (112, 4, 0, 32767),
    'Bus_Watchdog': (98, 1, 0, 127),
    'Goal_Position': (116, 4, -8191, 8191),
    'Torque_Enable': (64, 1, 0, 1),
}
GOAL_ADDRESS, GOAL_WIDTH, GOAL_MIN, GOAL_MAX = MOTION_REGISTERS['Goal_Position']


class FeedbackReadError(DynamixelCommunicationError):
    """只标记超时、坏包、截断等可恢复读取失败，不能用来忽略电机故障。"""


class _CommandCRCError(RuntimeError):
    """舵机明确拒绝了校验失败的指令，且未设置硬件告警位。"""


class KochTeleopBus(KochCalibrationBus):
    def __init__(self, port, motors, arm):
        super().__init__(port, motors)
        if arm not in ('leader', 'follower'):
            raise ValueError(arm)
        self.arm = arm
        # 各关节读取最终失败的次数（每次已含父类的 3 次重试），用于退出时定位坏线。
        self.read_failures = {}

    def _check(self, comm, error, operation):
        if comm == COMM_SUCCESS and error == ERRNUM_CRC:
            raise _CommandCRCError(f'{self.port} {operation}: {self._packet.getRxPacketError(error)}')
        super()._check(comm, error, operation)

    def _read_register(self, register, name, address, width):
        try:
            # 校准接口保持原样；运动接口将最终读失败分类，以便暂停轨迹后恢复。
            operation = f'读取 ID {self.motors[name][0]} {name}.{register}'
            for attempt in range(3):
                result = COMM_RX_CORRUPT
                try:
                    data, comm, error = self._packet.readTxRx(
                        self._port, self.motors[name][0], address, width)
                except IndexError:
                    self._port.is_using = False
                    reason = 'SDK 收到不完整状态包'
                else:
                    if comm in (COMM_RX_TIMEOUT, COMM_RX_CORRUPT):
                        result, reason = comm, self._packet.getTxRxResult(comm)
                    elif comm == COMM_SUCCESS and error == ERRNUM_CRC:
                        reason = self._packet.getRxPacketError(error)
                    else:
                        self._check(comm, error, operation)
                        if len(data) == width:
                            return int.from_bytes(bytes(data), 'little')
                        reason = f'状态数据长度 {len(data)}，预期 {width} 字节'
                if attempt < 2:
                    self._port.is_using = False
                    self._port.clearPort()
                    time.sleep(.01)
            raise FeedbackReadError(result, f'{self.port} {operation}: 连续 3 次读取失败；{reason}')
        except RuntimeError:
            self.read_failures[name] = self.read_failures.get(name, 0) + 1
            raise

    def read_motion(self, register, name):
        self._require_connection(False)
        address, width, _, _ = MOTION_REGISTERS[register]
        value = self._read_register(register, name, address, width)
        if register == 'Goal_Position' and value >= 2**31:
            value -= 2**32
        return value

    def read_joint_status(self, name):
        """用两个连续地址段读取原有健康检查项，保留长度校验和有限重试。"""
        self._require_connection(False)
        config = self._read_register('运行配置块', name, 10, 14).to_bytes(14, 'little')
        state = self._read_register('运行状态块', name, 64, 83).to_bytes(83, 'little')
        return {
            'Drive_Mode': config[0], 'Operating_Mode': config[1],
            'Homing_Offset': int.from_bytes(config[10:14], 'little', signed=True),
            'Torque_Enable': state[0], 'Hardware_Error_Status': state[6],
            'Bus_Watchdog': state[34],
            'Present_Input_Voltage': int.from_bytes(state[80:82], 'little'),
            'Present_Temperature': state[82],
        }

    def read_fault_snapshot(self, name):
        """只读停机诊断。允许 Alert 位以取得告警原因；正常控制仍拒绝它。"""
        self._require_connection(False)
        motor_id = self.motors[name][0]
        result = self._packet.readTx(self._port, motor_id, 64, 83)
        if result != COMM_SUCCESS:
            self._check(result, 0, '读取故障详情')
        packet, result = self._packet.rxPacket(self._port, False)
        if result != COMM_SUCCESS:
            self._check(result, 0, '读取故障详情')
        if len(packet) != 94 or packet[PKT_ID] != motor_id or packet[PKT_ERROR] & 0x7f:
            raise RuntimeError(f'ID {motor_id} 故障详情应答无效')
        start = PKT_PARAMETER0 + 1
        data = packet[start:start + 83]
        return {'fault': data[6], 'temperature': data[82],
                'voltage': int.from_bytes(bytes(data[80:82]), 'little') / 10,
                'torque': data[0]}

    def write_motion(self, register, name, value):
        self._require_connection(False)
        if self.arm != 'follower':
            raise ValueError('遥操作只读取主臂，禁止向主臂写入运动配置')
        address, width, low, high = MOTION_REGISTERS[register]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'{register} 必须在 {low}..{high} 内')
        if register == 'Profile_Velocity' and value == 0 and self.read('Torque_Enable', name) != 0:
            raise ValueError('速度 0 代表无限速，只能在扭矩关闭后恢复原配置')
        try:
            self._write_register(register, name, address, width, value)
        except _CommandCRCError as exc:
            if register != 'Torque_Enable':
                raise
            try:
                actual = self.read_motion(register, name)
                fault = self.read('Hardware_Error_Status', name)
            except RuntimeError as read_error:
                raise RuntimeError(f'{exc}；无法确认扭矩和健康状态：{read_error}') from read_error
            if fault or actual not in (0, 1):
                raise RuntimeError(f'{exc}；故障码 0x{fault:02X}，扭矩回读 {actual}，停止。') from exc
            if actual != value:
                # 仅在回读确认健康且写入未生效时，交给现有的有限重试。
                raise DynamixelCommunicationError(
                    COMM_RX_CORRUPT, f'{exc}；扭矩回读 {actual}，预期 {value}。') from exc

    def sync_write_goals(self, goals):
        """用一个广播 Sync Write 包写入全部关节目标，不等待任何应答。

        跟随目标是每周期都会被新目标取代的流数据：单包丢失只等于少发一帧，
        下一周期的新目标自然覆盖它，因此这里既不重发旧目标，也不因丢包停止。
        返回即表示数据已交给串口，不表示电机已执行；执行情况由位置回读和
        电机健康检查确认。
        """
        self._require_connection(False)
        if self.arm != 'follower':
            raise ValueError('遥操作只读取主臂，禁止向主臂写入运动配置')
        if set(goals) != set(self.motors):
            raise ValueError('必须一次写入全部关节的目标')
        param = bytearray()
        for name, value in goals.items():
            if type(value) is not int or not GOAL_MIN <= value <= GOAL_MAX:
                raise ValueError(f'Goal_Position 必须在 {GOAL_MIN}..{GOAL_MAX} 内')
            param.append(self.motors[name][0])
            param += (value & 0xFFFFFFFF).to_bytes(GOAL_WIDTH, 'little')
        result = self._packet.syncWriteTxOnly(self._port, GOAL_ADDRESS, GOAL_WIDTH,
                                              bytes(param), len(param))
        if result != COMM_SUCCESS:
            raise DynamixelCommunicationError(
                result, f'{self.port} 同步写入 Goal_Position: {self._packet.getTxRxResult(result)}')
