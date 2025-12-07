import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


@dataclass
class ClientConfig:
    host: str
    port: int
    vllm_api_key: str
    max_tokens: int = 512
    temperature: float = 0.4
    system_prompt: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/"


class GatewayClient:
    def __init__(self, config: ClientConfig):
        self.config = config
        self.session_id: Optional[str] = None

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        target = urljoin(self.config.base_url, path.lstrip("/"))
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            target,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:  # noqa: BLE001
            error_body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code} at {target}: {error_body}") from exc
        except URLError as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to reach {target}: {exc.reason}") from exc

    def init_session(self) -> str:
        payload = {"vllm_api_key": self.config.vllm_api_key}
        response = self._post("/init_session", payload)
        self.session_id = response.get("session_id")
        if not self.session_id:
            raise RuntimeError("Gateway did not return session_id")
        return self.session_id

    def set_system_prompt(self, system_prompt: str) -> None:
        if not self.session_id:
            raise RuntimeError("Call init_session() before setting system prompt")
        self._post(
            "/set_system_prompt",
            {"session_id": self.session_id, "system_prompt": system_prompt},
        )

    def chat(self, message: str) -> str:
        if not self.session_id:
            raise RuntimeError("Call init_session() before chatting")
        payload: Dict[str, Any] = {
            "session_id": self.session_id,
            "message": message,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if self.config.extra:
            payload["extra"] = self.config.extra
        response = self._post("/chat", payload)
        reply = response.get("reply", "")
        finish_reason = response.get("finish_reason")
        tokens_info = self._format_tokens(response)
        if finish_reason:
            return f"{reply}\n[finish_reason={finish_reason}{tokens_info}]"
        if tokens_info:
            return f"{reply}\n{tokens_info}"
        return reply

    @staticmethod
    def _format_tokens(response: Dict[str, Any]) -> str:
        parts = []
        for field in ("total_tokens", "input_tokens", "output_tokens"):
            if response.get(field) is not None:
                parts.append(f"{field}={response[field]}")
        return f"({', '.join(parts)})" if parts else ""


def load_config(path: Path) -> ClientConfig:
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    missing = [key for key in ("host", "port", "vllm_api_key") if key not in raw]
    if missing:
        raise ValueError(f"Missing required fields in config: {', '.join(missing)}")
    return ClientConfig(
        host=str(raw["host"]),
        port=int(raw["port"]),
        vllm_api_key=str(raw["vllm_api_key"]),
        max_tokens=int(raw.get("max_tokens", 512)),
        temperature=float(raw.get("temperature", 0.4)),
        system_prompt=raw.get("system_prompt"),
        extra=raw.get("extra"),
    )


def ask_for_system_prompt(default_prompt: Optional[str]) -> Optional[str]:
    if default_prompt:
        use_default = input(
            "Использовать системный промпт из конфигурации? [Y/n]: "
        ).strip().lower()
        if use_default in {"", "y", "yes"}:
            return default_prompt
    user_prompt = input(
        "Введите системный промпт (оставьте пустым, чтобы оставить стандартный): "
    ).strip()
    return user_prompt or None


def chat_loop(client: GatewayClient) -> None:
    print("Введите сообщение. /system <текст> — сменить системный промпт, /exit — выйти.")
    try:
        while True:
            user_input = input("Вы: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"/exit", ":q", "/quit"}:
                print("Выход из чата...")
                break
            if user_input.startswith("/system"):
                _, _, new_prompt = user_input.partition(" ")
                new_prompt = new_prompt.strip()
                if not new_prompt:
                    print("Укажите новый системный промпт после команды /system")
                    continue
                client.set_system_prompt(new_prompt)
                print("Системный промпт обновлён.")
                continue
            try:
                reply = client.chat(user_input)
            except Exception as exc:  # noqa: BLE001
                print(f"Ошибка во время запроса: {exc}")
                continue
            print(f"Модель: {reply}")
    except KeyboardInterrupt:
        print("\nВыход из чата по Ctrl+C...")


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI клиент для Simple Chat Gateway")
    default_config = Path(__file__).with_name("chat.example.json")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"Путь до JSON конфигурации (по умолчанию {default_config})",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    client = GatewayClient(config)

    try:
        session_id = client.init_session()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Не удалось создать сессию: {exc}")

    print(f"Сессия создана: {session_id}. TTL определяется настройками gateway.")

    system_prompt = ask_for_system_prompt(config.system_prompt)
    if system_prompt:
        try:
            client.set_system_prompt(system_prompt)
            print("Системный промпт установлен.")
        except Exception as exc:  # noqa: BLE001
            print(f"Не удалось установить системный промпт: {exc}")

    chat_loop(client)


if __name__ == "__main__":
    main()
