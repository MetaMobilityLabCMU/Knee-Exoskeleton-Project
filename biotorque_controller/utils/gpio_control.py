import Jetson.GPIO as GPIO


class TrialGPIO:
    def __init__(self, pin: int):
        self.pin = int(pin)
        self._state = 0
        try:
            GPIO.setwarnings(False)
            GPIO.cleanup()
        except Exception:
            pass
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
        self._first_pulse_sent = False
        self._first_pulse_end = None
        self._second_pulse_sent = False
        self._second_pulse_end = None

    def pulse_start(self):
        GPIO.output(self.pin, GPIO.HIGH)
        self._state = 1

    def pulse_end(self):
        GPIO.output(self.pin, GPIO.LOW)
        self._state = 0

    def state(self) -> int:
        return self._state

    def update_trial_pulses(
        self,
        elapsed: float,
        start_delay: float,
        trial_duration: float,
        pulse_width: float,
    ) -> None:
        if not self._first_pulse_sent and elapsed >= start_delay:
            try:
                self.pulse_start()
            except Exception as error:
                print(f"[GPIO] first pulse_start error: {error}")
            self._first_pulse_sent = True
            self._first_pulse_end = elapsed + pulse_width

        if self._first_pulse_end is not None and elapsed >= self._first_pulse_end:
            try:
                self.pulse_end()
            except Exception as error:
                print(f"[GPIO] first pulse_end error: {error}")
            self._first_pulse_end = None

        if not self._second_pulse_sent and elapsed >= trial_duration:
            try:
                self.pulse_start()
            except Exception as error:
                print(f"[GPIO] second pulse_start error: {error}")
            self._second_pulse_sent = True
            self._second_pulse_end = elapsed + pulse_width

        if self._second_pulse_end is not None and elapsed >= self._second_pulse_end:
            try:
                self.pulse_end()
            except Exception as error:
                print(f"[GPIO] second pulse_end error: {error}")
            self._second_pulse_end = None

    def close(self):
        try:
            GPIO.output(self.pin, GPIO.LOW)
            GPIO.cleanup()
        except Exception:
            pass
