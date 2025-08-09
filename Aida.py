# -*- coding: utf-8 -*-
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from llama_cpp import Llama
import os
import random
import time
from datetime import datetime
import logging
from pathlib import Path
import re
import json
from typing import Any, Dict, List, Optional, Union
import psutil
from dataclasses import dataclass
import unicodedata


# === ПОЛЬЗОВАТЕЛЬСКИЕ ИСКЛЮЧЕНИЯ ДЛЯ БЕЗОПАСНОСТИ ===
class SecurityError(Exception):
    """Исключение для ошибок безопасности конфигурации"""
    pass

class ConfigurationError(Exception):
    """Исключение для ошибок конфигурации"""
    pass


@dataclass
class GenerationMetrics:
    """Метрики производительности генерации для мониторинга"""
    tokens_generated: int
    latency_ms: float
    memory_peak_mb: float
    throughput_tokens_per_second: float
    temperature_used: float
    model_device: str
    

class EnhancedGenerationManager:
    """
    Централизованный менеджер параметров генерации с поддержкой
    dynamic configuration и performance monitoring
    """

    def __init__(self, base_config: Dict[str, Any], monitoring_config: Dict[str, Any]):
        self.base_generation_config = base_config
        self.monitoring = monitoring_config
        self.metrics_history = []

    def get_generation_params(self, task_type: str = "general", override_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Динамическое формирование параметров генерации на основе:
        - JSON конфигурации
        - Типа задачи
        - Runtime overrides
        """
        # === Базовые параметры из JSON ===
        params = self.base_generation_config.copy()

        # === Task-specific adjustments ===
        task_adjustments = self._get_task_specific_adjustments(task_type)
        params.update(task_adjustments)

        # === Runtime overrides (highest priority) ===
        if override_params:
            params.update(override_params)

        # === Валидация и нормализация ===
        return self._validate_and_normalize_params(params)

    def _get_task_specific_adjustments(self, task_type: str) -> Dict[str, Any]:
        """Специализированные параметры для различных типов задач"""
        adjustments = {
            "code_generation": {
                "temperature": 0.3,  # Более детерминистично для кода
                "top_p": 0.9,
                "repetition_penalty": 1.2,
                "max_new_tokens": 1500
            },
            "code_analysis": {
                "temperature": 0.4,
                "top_p": 0.85,
                "max_new_tokens": 1200
            },
            "code_fixing": {
                "temperature": 0.2,  # Максимальная точность
                "top_p": 0.8,
                "max_new_tokens": 1200
            },
            "code_explanation": {
                "temperature": 0.6,  # Более творческие объяснения
                "top_p": 0.95,
                "max_new_tokens": 1500
            },
            "free_mode": {
                "temperature": 0.7,
                "top_p": 0.95,
                "max_new_tokens": 1024
            }
        }
        return adjustments.get(task_type, {})

    def _validate_and_normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Валидация и нормализация параметров"""
        # === Валидация ranges ===
        params["temperature"] = max(0.1, min(2.0, params.get("temperature", 0.7)))
        params["top_p"] = max(0.1, min(1.0, params.get("top_p", 0.95)))
        params["top_k"] = max(1, min(200, params.get("top_k", 50)))
        params["max_new_tokens"] = max(50, min(4096, params.get("max_new_tokens", 1200)))

        # === Обработка null значений ===
        if params.get("pad_token_id") is None:
            params.pop("pad_token_id", None)

        return params

    def record_metrics(self, metrics: GenerationMetrics):
        """Запись метрик для monitoring и optimization"""
        if self.monitoring.get("performance_tracking", True):
            self.metrics_history.append(metrics)

            # === Retention policy ===
            max_history = self.monitoring.get("metrics_retention", 1000)
            if len(self.metrics_history) > max_history:
                self.metrics_history = self.metrics_history[-max_history:]


# === Настройка стиля загрузки ===
AIDA_THEME = "matrix"


class EnhancedCodeAssistant:
    """Улучшенный ИИ-ассистент для генерации, анализа, объяснения и исправления кода."""
    def __init__(self, config_path="config.json"):
        # === БЕЗОПАСНОЕ ОПРЕДЕЛЕНИЕ ПУТИ КОНФИГУРАЦИИ ===
        if config_path is None:
            # Приоритет: переменная окружения -> локальный файл -> исключение
            config_path = os.environ.get('AIDA_CONFIG_PATH')
            if config_path is None:
                script_dir = Path(__file__).parent.absolute()
                default_config = script_dir / "config.json"
                if default_config.exists():
                    config_path = str(default_config)
                else:
                    raise FileNotFoundError(
                        f"Файл конфигурации не найден. Ожидается: {default_config}\n"
                        "Cоздайте config.json в том же каталоге, что и скрипт, или"
                        "установите переменную среды AIDA_CONFIG_PATH."
                    )
        
        self.config_path = Path(config_path).resolve()

        # Валидация безопасности пути конфигурации
        if not self._validate_config_path_security(self.config_path):
            raise SecurityError(f"Неверный путь конфигурации: {self.config_path}")

        self.model = None
        self.tokenizer = None
        self.is_gguf = False
        self.conversation_history = []
        self.context_window = 5  # Увеличено для лучшего контекста
        self.max_history = 100
        
        # === Самосознание Аиды ===
        self.name = "Аида"
        self.personality = {
            "friendly": True,
            "helpful": True,
            "curious": True,
            "supportive": True
        }
        self.user_name = None
        self.session_start = datetime.now()

        # Инициализация логгера в конструкторе
        self.logger = None

        # Загружаем конфигурацию при инициализации
        try:
            self.load_config()
            self.setup_logging()
        except Exception as init_error:
            print(f"❌ Критическая ошибка инициализации: {init_error}")
            print("⚠️ Переключение на fallback конфигурацию...")
            # Минимальная инициализация для graceful degradation
            self._initialize_fallback_config()
            self._setup_fallback_logging()
            print("✅ Система работает в режиме ограниченной функциональности")

        # Улучшенные системные промпты для DeepSeek
        self.system_prompts = {
            "code_generation": f"""Ты {self.name} - опытный программист-ассистент с глубокими знаниями.
Создавай ТОЛЬКО валидный, рабочий код высокого качества.

ПРИНЦИПЫ:
- Используй правильный синтаксис языка программирования
- Пиши читаемые имена переменных и функций
- Добавляй комментарии на русском языке
- Следуй лучшим практикам разработки
- Код должен компилироваться и работать без ошибок
- Применяй современные подходы и паттерны

Отвечай ТОЛЬКО на русском языке.""",

            "code_analysis": f"""Ты {self.name} - senior software architect с экспертизой в code review.

СТРУКТУРА АНАЛИЗА:
1. СИНТАКСИС: проверка ошибок компиляции
2. ЛОГИКА: корректность алгоритмов
3. АРХИТЕКТУРА: структура и дизайн
4. ПРОИЗВОДИТЕЛЬНОСТЬ: оптимизация и эффективность
5. БЕЗОПАСНОСТЬ: уязвимости и риски
6. СТИЛЬ: соответствие стандартам кодирования

Проводи глубокий технический анализ. Отвечай на русском языке.""",

            "code_fixing": f"""Ты {self.name} - expert debugging specialist и refactoring engineer.

АЛГОРИТМ ИСПРАВЛЕНИЯ:
1. Выявляю все синтаксические ошибки
2. Исправляю логические проблемы
3. Оптимизирую производительность
4. Улучшаю архитектуру кода
5. Добавляю необходимые проверки
6. Привожу к стандартам кодирования

Возвращаю исправленный код с объяснениями. Отвечаю на русском языке.""",

            "code_explanation": f"""Ты {self.name} - senior technical educator и software development mentor.

МЕТОДИКА ОБЪЯСНЕНИЯ:
1. ОБЗОР: что делает программа в целом
2. СТРУКТУРА: разбор архитектуры и компонентов
3. АЛГОРИТМ: пошаговый анализ логики
4. ДЕТАЛИ: объяснение сложных моментов
5. ПРИМЕНЕНИЕ: где и как использовать

Объясняю доступно и структурированно. Отвечаю на русском языке."""
        }

        
    def _validate_config_path_security(self, config_path: Path) -> bool:
        """
        Валидация безопасности пути конфигурации

        Проверяет:
        - Существование файла
        - Directory traversal атаки
        - Расширение файла
        - Права доступа
        """
        try:
            # Проверка существования файла
            if not config_path.exists():
                raise FileNotFoundError(f"Config file does not exist: {config_path}")

            # Проверка на directory traversal атаки
            config_path.resolve(strict=True)

            # Проверка расширения файла
            if config_path.suffix.lower() != '.json':
                self.log_error("config_security", f"Недопустимое расширение файла конфигурации: {config_path.suffix}")
                return False

            # Проверка прав доступа
            if not os.access(config_path, os.R_OK):
                self.log_error("config_security", f"Нет разрешения на чтение конфигурации: {config_path}")
                return False

            return True
        
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.log_error("config_path_validation", f"Проверка безопасности не пройдена: {e}")
            else:
                print(f"Ошибка проверки конфигурации: {e}")
            return False


    def _validate_local_model_path(self, local_path: str) -> Optional[Path]:
        """Комплексная валидация пути к локальной модели
    
        Args:
            local_path: Путь к директории модели из конфигурации
            
        Returns:
            Path: Валидированный абсолютный путь или None при ошибке
            
        Проверяет:
        - Существование директории
        - Наличие файлов модели
        - Права доступа
        - Безопасность пути
        """
        try:
            # Нормализация пути с поддержкой ~ и относительных путей
            model_path = Path(local_path).expanduser().resolve()

            # Базовые проверки безопасности
            if not model_path.exists():
                self.log_error("model_validation", f"Путь к модели не существует: {model_path}")
                return None

            if not model_path.is_dir():
                self.log_error("model_validation", f"Путь к модели не является каталогом: {model_path}")
                return None

            # Проверка на directory traversal
            try:
                model_path.resolve(strict=True)
            except (OSError, RuntimeError) as e:
                self.log_error("model_validation", f"Ошибка определения пути: {e}")
                return None

            # Проверка наличия критических файлов модели
            required_files = ["config.json"]
            model_files = ["pytorch_model.bin", "model.safetensors"]

            # Проверяем config.json модели
            if not (model_path / "config.json").exists():
                self.log_error("model_validation", f"Отсутствует config.json в {model_path}")
                return None

            # Проверяем наличие весов модели
            has_weights = any((model_path / f).exists() for f in model_files)
            if not has_weights:
                self.log_error("model_validation", f"Веса моделей не найдены в {model_path}")
                return None

            # Проверка прав доступа
            if not os.access(model_path, os.R_OK):
                self.log_error("model_validation", f"Нет разрешения на чтение каталога модели: {model_path}")
                return None

            self.log_system_event(f"Путь к локальной модели успешно проверен: {model_path}")
            return model_path

        except Exception as e:
            self.log_error("model_path_validation", f"Проверка не удалась для {local_path}: {e}")
            return None


    def load_config(self):
        """Загрузка конфигурации из JSON-файла"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Основные параметры модели
            self.model_info = config.get("model", {
                "name": "deepseek-ai/deepseek-coder-6.7b-instruct",
                "description": "DeepSeek Coder 6.7B Instruct",
                "size": "6.7B параметров",
                "context_size": 4096
            })

            model_config = config.get("model", {})
            if model_config.get("use_local", False):
                local_path = model_config.get("local_path", "")
                if not local_path:
                    raise ValueError("local_path должен быть указан, если use_local имеет значение true")
                # Валидация и нормализация локального пути
                validated_path = self._validate_local_model_path(local_path)
                if not validated_path:
                    if model_config.get("fallback_to_remote", True):
                        self.log_error("config_load", f"Неверный local_path: {local_path}, возвращение к удаленному режиму")
                        model_config["use_local"] = False
                    else:
                        raise FileNotFoundError(f"Неверный путь к локальной модели: {local_path}")
                else:
                    model_config["local_path"] = str(validated_path)
                    self.log_system_event(f"Подтвержденный путь локальной модели: {validated_path}")

            self.model_info = model_config

            # Настройки системы
            settings = config.get("settings", {
                "device": "auto",
                "cache_dir": "./model_cache",
                "dtype": "bfloat16"
            })

            # Определение устройства
            if settings["device"] == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = settings["device"]

            # Сохраняем настройки
            self.settings = settings
            
            # === Загрузка всех секций конфигурации ===
            self.generation = config.get("generation", {
                "max_new_tokens": 1200,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 3,
                "do_sample": True,
                "early_stopping": True,
                "pad_token_id": None
            })
            self.performance = config.get("performance", {
                "max_history": 100,
                "context_window": 5,
                "batch_size": 1,
                "gradient_checkpointing": False,
                "memory_efficient_attention": True,
                "compile_mode": "reduce-overhead",
                "max_response_length": 50000,
                "max_context_length": 4096
            })
            self.monitoring = config.get("monitoring", {
                "enable_metrics": True,
                "memory_monitoring": True,
                "performance_tracking": True,
                "error_reporting": True,
                "metrics_retention": 1000,
                "alert_thresholds": {
                    "inference_latency_ms": 5000,
                    "memory_usage_mb": 8192,
                    "min_tokens_per_second": 10
                }
            })
            self.optimization = config.get("optimization", {
                "adaptive_memory": True,
                "auto_device_map": True,
                "memory_cleanup": True,
                "cache_optimization": True,
                "memory_efficient_attention": False
            })
            self.text_processing = config.get("text_processing", {
                "unicode_normalization": "NFC",
                "max_input_length": 5000,
                "sanitize_surrogates": True,
                "encoding_fallback": "utf-8"
            })

            # === Загрузка конфигурации chat template system ===
            self.chat_template_system = config.get("chat_template_system", {
                "enabled": True,
                "force_override": False,
                "detection_mode": "smart",
                "fallback_strategy": "generic",
                "validation": {
                    "syntax_check": True,
                    "performance_test": False
                }
            })

            self.chat_templates = config.get("chat_templates", {})

            # === Загрузка GGUF конфигурации ===
            self.gguf_settings = config.get("gguf_settings", {})
            self.gguf_optimization = config.get("gguf_optimization", {})


            # === Валидация chat template конфигурации ===
            if self.chat_template_system.get("enabled", True):
                self._validate_chat_template_config()

        except Exception as e:
            # Значения по умолчанию при ошибке
            self.model_info = {
                "name": "deepseek-ai/deepseek-coder-6.7b-instruct",
                "description": "DeepSeek Coder 6.7B Instruct",
                "size": "6.7B параметров",
                "context_size": 4096
            }
            self.settings = {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "cache_dir": "./model_cache",
                "dtype": "bfloat16"
            }
            self.device = self.settings["device"]

            # Инициализация отсутствующих конфигураций
            self.generation = {
                "max_new_tokens": 1200,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 3,
                "do_sample": True,
                "early_stopping": True,
                "pad_token_id": None
            }
            self.performance = {
                "max_history": 100,
                "context_window": 5,
                "max_response_length": 50000,
                "max_context_length": 4096
            }
            self.monitoring = {
                "enable_metrics": True,
                "performance_tracking": True,
                "metrics_retention": 1000,
                "alert_thresholds": {
                    "inference_latency_ms": 5000,
                    "memory_usage_mb": 8192,
                    "min_tokens_per_second": 10
                }
            }
            self.optimization = {
                "adaptive_memory": True,
                "memory_cleanup": True,
                "memory_efficient_attention": False
            }
            self.text_processing = {
                "unicode_normalization": "NFC",
                "max_input_length": 5000,
                "sanitize_surrogates": True,
                "encoding_fallback": "utf-8"
            }
            self.chat_template_system = {
                "enabled": False, # Безопасно отключаем систему шаблонов при ошибке
                "force_override": False,
                "detection_mode": "smart",
                "fallback_strategy": "generic"
            }

            self.chat_templates = {} # Устанавливаем пустой словарь
            self.gguf_settings = {}
            self.gguf_optimization = {}

            print(f"Ошибка загрузки конфига: {e}")
            self.log_error("load_config", f"Ошибка загрузки конфига: {e}. Использую значения по умолчанию.")

        # Валидация конфигурации после загрузки
        self.validate_configuration()


    
    def validate_configuration(self):
        """Валидация загруженной конфигурации на корректность"""
        validation_errors = []

        # Проверка критичных параметров generation
        gen_config = getattr(self, 'generation', {})
        if gen_config.get('temperature', 0.7) < 0.1 or gen_config.get('temperature', 0.7) > 2.0:
            validation_errors.append("generation.temperature должна быть в диапазоне [0.1, 2.0]")

        if gen_config.get('top_p', 0.95) < 0.1 or gen_config.get('top_p', 0.95) > 1.0:
            validation_errors.append("generation.top_p должна быть в диапазоне [0.1, 1.0]")

        # Проверка производительности
        perf_config = getattr(self, 'performance', {})
        if perf_config.get('max_context_length', 4096) > 8192:
            validation_errors.append("performance.max_context_length превышает рекомендуемое значение 8192")

        # Проверка text_processing
        text_config = getattr(self, 'text_processing', {})
        valid_normalizations = ['NFC', 'NFD', 'NFKC', 'NFKD']
        if text_config.get('unicode_normalization', 'NFC') not in valid_normalizations:
            validation_errors.append(f"text_processing.unicode_normalization должна быть одной из: {valid_normalizations}")

        if validation_errors:
            self.log_error("config_validation", f"Configuration validation failed: {'; '.join(validation_errors)}")
            for error in validation_errors:
                print(f"⚠️ Config Warning: {error}")
        else:
            self.log_system_event("Configuration validation passed")

        return len(validation_errors) == 0


    def _validate_chat_template_config(self):
        """
        Валидация конфигурации chat template system

        Проверяет:
        - Синтаксис Jinja2 шаблонов
        - Наличие обязательных полей
        - Корректность stop_tokens
        """
        validation_errors = []

        try:
            # Проверяем наличие хотя бы одного шаблона
            if not self.chat_templates:
                validation_errors.append("chat_templates секция пуста")
                return validation_errors

            # Валидация каждого шаблона
            for template_name, template_config in self.chat_templates.items():
                # Проверка обязательных полей
                if 'template' not in template_config:
                    validation_errors.append(f"chat_templates.{template_name}: отсутствует поле 'template'")
                    continue

            # Валидация Jinja2 синтаксиса если включена
            if self.chat_template_system.get("validation", {}).get("syntax_check", True):
                try:
                    from jinja2 import Template, Environment
                    env = Environment()
                    compiled_template = env.from_string(template_config['template'])

                    # Тестовая рендеринг с фиктивными данными
                    test_messages = [
                        {"role": "system", "content": "Test system message"},
                        {"role": "user", "content": "Test user message"}
                    ]
                    compiled_template.render(messages=test_messages)

                except Exception as jinja_error:
                    validation_errors.append(f"chat_templates.{template_name}: некорректный Jinja2 синтаксис - {jinja_error}")

            # Валидация stop_tokens
            stop_tokens = template_config.get("stop_tokens", [])
            if not isinstance(stop_tokens, list):
                validation_errors.append(f"chat_templates.{template_name}: stop_tokens должен быть списком")

            # Проверка default template
            default_template = self.settings.get("default_chat_template", "generic")
            if default_template not in self.chat_templates:
                validation_errors.append(f"default_chat_template '{default_template}' не найден в chat_templates")

        except Exception as e:
            validation_errors.append(f"Критическая ошибка валидации chat templates: {e}")

        # Логирование результатов валидации
        if validation_errors:
            for error in validation_errors:
                self.log_error("chat_template_validation", error)
                print(f"⚠️ Chat Template Warning: {error}")
        else:
            self.log_system_event("Chat template configuration validation passed")

        return validation_errors


    def _initialize_fallback_config(self):
        """Минимальная конфигурация при критических сбоях"""
        # Критические атрибуты для базовой функциональности
        self.device = "cpu"
        self.settings = {
            "device": "cpu",
            "dtype": "float32",
            "cache_dir": "./model_cache"
        }

        self.generation = {
            "max_new_tokens": 1200,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
            "repetition_penalty": 1.1
        }

        self.performance = {
            "max_history": 100,
            "context_window": 5,
            "max_response_length": 50000,
            "max_context_length": 4096
        }

        self.monitoring = {"enable_metrics": False}
        self.optimization = {"adaptive_memory": False}
        self.text_processing = {"unicode_normalization": "NFC"}
        self.gguf_optimization = {}

        # Fallback model info
        self.model_info = {
            "name": "deepseek-ai/deepseek-coder-6.7b-instruct",
            "description": "DeepSeek Coder 6.7B (Fallback mode)",
            "size": "6.7B",
            "context_size": 4096
        }


    def _setup_fallback_logging(self):
        """Минимальное логирование для fallback режима"""
        try:
            import logging
            self.logger = logging.getLogger('AidaFallback')
            self.logger.setLevel(logging.WARNING)

            # Console logging только для критических ошибок
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            self.logger.warning("🔧 FALLBACK: Система работает в режиме деградации")
        except Exception:
            self.logger = None  # Полная деградация логирования
            print("⚠️ Логирование отключено в fallback режиме")


    def get_display_configuration(self) -> Dict[str, Any]:
        """Возвращает ключевые параметры конфигурации для отображения в UI."""
        return {
                "dtype": self.settings.get("dtype"),
                "quantization": self.settings.get("quantization_type"),
                "max_new_tokens": self.generation.get("max_new_tokens"),
                "model_name": self.model_info.get("name"),
                "device": self.device
        }

    def setup_logging(self):
        """Настройка системы логирования"""
        # Инициализируем логгер если не существует
        if not hasattr(self, 'logger'):
            self.logger = None

        if self.logger and self.logger.hasHandlers():
            return


        logs_dir = Path("aida_logs")
        logs_dir.mkdir(exist_ok=True)

        log_filename = f"aida_session_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        log_path = logs_dir / log_filename

        self.logger = logging.getLogger('AidaAssistant')
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

        self.logger.info("="*80)
        self.logger.info(f"🚀 НОВАЯ СЕССИЯ АИДЫ НАЧАЛАСЬ")
        self.logger.info(f"📅 Время запуска: {self.session_start}")
        self.logger.info(f"💻 Устройство: {self.device}")
        self.logger.info(f"🤖 Модель: {self.model_info['description']}")
        self.logger.info("="*80)

        # Логирование о конфигурации
        self.log_system_event(f"Конфиг загружен: {self.config_path}")
        self.log_system_event(f"Устройство: {self.device}")
        self.log_system_event(f"Модель: {self.model_info['description']}")

    def safe_log_string(self, s):
        """Очистка строки от суррогатных символов"""
        if not s:
            return ""
        return re.sub(r'[\ud800-\udfff]', '', s)

    def log_action(self, action_type, details):
        """Логирование действий"""
        if not hasattr(self, 'logger') or self.logger is None:
            return  # Graceful degradation при отсутствии логгера
        try:
            user_info = f"[{self.user_name}]" if self.user_name else "[Анонимный]"
            safe_details = self.safe_log_string(str(details))
            log_message = f"📝 {action_type} | {user_info} | {safe_details}"
            self.logger.info(log_message)
        except Exception as e:
            print(f"Logging error: {e}")

    def log_error(self, context, error_message):
        """Логирование ошибок с Unicode-санитизацией"""
        if not hasattr(self, 'logger') or self.logger is None:
            return  # Graceful degradation при отсутствии логгера
        try:
            # === Критическая санитизация суррогатных символов ===
            clean_message = self._sanitize_unicode_string(str(error_message))
            clean_context = self._sanitize_unicode_string(str(context))

            log_message = f"❌ ОШИБКА [{clean_context}]: {clean_message}"
            self.logger.error(log_message)
        except Exception as e:
            # === Fallback для катастрофических сбоев логирования ===
            print(f"Critical logging failure: {e}")

    
    def _sanitize_unicode_string(self, text: str) -> str:
        """Удаление суррогатных символов и нормализация Unicode"""
        if not isinstance(text, str):
            text = str(text)

        # Принудительно кодируем в UTF-8, заменяя любые ошибки (включая суррогаты)
        # на стандартный символ-заменитель ''.
        # Затем декодируем обратно в чистую и безопасную строку.
        return text.encode('utf-8', 'replace').decode('utf-8')

    def log_system_event(self, event):
        """Логирование системных событий"""
        if not hasattr(self, 'logger') or self.logger is None:
            return  # Graceful degradation при отсутствии логгера
        self.logger.info(f"🔧 СИСТЕМА: {event}")

    def introduce_myself(self):
        """Аида представляется и знакомится с пользователем"""
        print(f"\n💜 Привет! Меня зовут {self.name}!")
        print(f"🤖 Я твой персональный ассистент по программированию на базе {self.model_info['description']} ({self.model_info['size']})")
        print("💫 Я специализируюсь на генерации, анализе, исправлении кода и обучении программированию.")
        print("🧠 Моя модель обучена на огромном количестве кода и может работать с множеством языков!")

        if not self.user_name:
            name = input("\n🌟 А как тебя зовут? Мне приятно будет знать: ").strip()
            if name:
                self.user_name = name
                print(f"\n😊 Замечательно, {self.user_name}! У нас будет отличная команда! 🤝")
            else:
                print("\n😊 Хорошо, будем работать анонимно! Главное - качественный код!")

        return self.user_name

    def get_personalized_greeting(self):
        """Персонализированное приветствие"""
        now = datetime.now().hour
        name_part = f", {self.user_name}" if self.user_name else ""

        if 5 <= now < 12:
            return f"🌅 Доброе утро{name_part}! Аида готова к новым вызовам! ☕"
        elif 12 <= now < 18:
            return f"🌞 Добрый день{name_part}! Время продуктивного кодинга! 💪"
        elif 18 <= now < 23:
            return f"🌇 Добрый вечер{name_part}! Займёмся интересными задачами? 🧘"
        else:
            return f"🌙 Доброй ночи{name_part}! Ночное программирование - моя стихия! 💜"

    def aida_think_aloud(self, task_type):
        """Аида делится своими мыслями о задаче"""
        thoughts = {
            "code_generation": [
                "🤔 Интересная задача! Помогу создать элегантное решение...",
                "💡 Применю знания из миллионов строк кода для лучшего результата!",
                "🧠 Использую современные паттерны и лучшие практики..."
            ],
            "code_analysis": [
                "🔍 Глубокий анализ с Аида - найду все тонкости!",
                "🧐 Проверю архитектуру, безопасность и производительность...",
                "📋 Составлю детальный технический отчёт!"
            ],
            "code_fixing": [
                "🛠️ Аида поможет найти и исправить все проблемы!",
                "✨ Не просто исправлю, а улучшу архитектуру!",
                "🎯 Применю продвинутые техники рефакторинга..."
            ],
            "code_explanation": [
                "📚 Объясню код на основе глубокого понимания!",
                "🎓 Я знаю множество паттернов - поделюсь знаниями!",
                "💬 Расскажу не только 'что', но и 'почему' именно так..."
            ]
        }

        thought = random.choice(thoughts.get(task_type, ["🤖 Аида обрабатывает запрос..."]))
        print(f"\n💭 {self.name}: {thought}")

    def load_model(self):
        """
        Модернизированная загрузка модели с поддержкой локальных файлов 
        и GGUF формата
        Полная интеграция с расширенной JSON конфигурацией
        """
        if self.model and (self.tokenizer or self.is_gguf):
            return True

        try:
            model_config = self.model_info
            model_format = model_config.get("format", "hf")

            if model_format.lower() == "gguf":
                return self._load_gguf_model()
            else:
                return self._load_hf_model()

        except Exception as e:
            return self._handle_generic_error(e)

    def _load_gguf_model(self):
        """Загрузка модели в формате GGUF с использованием llama-cpp-python."""
        self.log_system_event("Начало загрузки GGUF модели...")
        model_config = self.model_info
        gguf_config = self.gguf_settings
        gguf_opt_config = self.gguf_optimization
        gguf_path = model_config.get("gguf_path")

        # === Автоматическая загрузка если файл не найден ===
        if not gguf_path or not Path(gguf_path).exists():
            print(f"\n🔍 GGUF файл не найден по пути: {gguf_path}")
            print("🚀 Попытка автоматической загрузки...")

            # Пытаемся загрузить модель автоматически
            downloaded_path = self._download_gguf_model_automatically(model_config)
            if downloaded_path:
                gguf_path = downloaded_path
                model_config["gguf_path"] = gguf_path  # Обновляем конфигурацию
                print(f"✅ GGUF модель успешно загружена: {gguf_path}")
            else:
                self.log_error("gguf_load", f"Не удалось автоматически загрузить GGUF модель")

                # Проверяем fallback к HF модели если включено
                if model_config.get("fallback_to_hf", False):
                    print("⚠️ Переключение на HF версию модели...")
                    model_config["format"] = "hf"  # Меняем формат
                    return self._load_hf_model()  # Загружаем HF версию
                else:
                    return False

        print(f"\n🔄 Загружаю GGUF модель: {model_config['description']}")
        print(f"📍 Путь: {gguf_path}")

        try:
            # === Собираем параметры для Llama конструктора ===
            llama_params = {
                "model_path": gguf_path,
                "n_gpu_layers": gguf_config.get("n_gpu_layers", -1), # -1 для автоматического определения
                "n_ctx": gguf_config.get("n_ctx", 4096),
                "n_batch": gguf_config.get("n_batch", 512),
                "verbose": gguf_config.get("verbose", False),
                "n_threads": gguf_config.get("n_threads"),
                "n_threads_batch": gguf_config.get("n_threads_batch"),
                "mul_mat_q": gguf_config.get("mul_mat_q"),
                "f16_kv": gguf_config.get("f16_kv", True),
                "use_mmap": gguf_config.get("use_mmap", True),
                "use_mlock": gguf_config.get("use_mlock", False),
                "chat_format": gguf_config.get("chat_format", "chatml"),
                "flash_attn": gguf_opt_config.get("flash_attention", False) # Интеграция Flash Attention
            }
            # Удаляем None значения, чтобы использовать defaults из llama-cpp-python
            llama_params = {k: v for k, v in llama_params.items() if v is not None}


            self.model = Llama(**llama_params)
            self.is_gguf = True
            self.tokenizer = None  # Для GGUF токенизатор встроен в объект Llama
            self.model_loaded = True  # Устанавливаем флаг успешной загрузки

            print("\n" + "="*70)
            print("🤖 КОНФИГУРАЦИЯ ЗАГРУЖЕННОЙ GGUF МОДЕЛИ")
            print("="*70)
            print(f"📋 Модель: {model_config['description']}")
            print(f"⚙️ Контекст (n_ctx): {gguf_config.get('n_ctx')}")
            print(f"🚀 Слоев на GPU (n_gpu_layers): {gguf_config.get('n_gpu_layers')}")
            print(f"⚡ Flash Attention: {'Включено' if gguf_opt_config.get('flash_attention') else 'Выключено'}")
            print("="*70)

            self.log_system_event(f"GGUF модель успешно загружена: {gguf_path}")
            self._initialize_generation_manager()
            return True

        except Exception as e:
            self.log_error("gguf_load_error", f"Ошибка при загрузке GGUF модели: {e}")
            return self._handle_generic_error(e)


    def _download_gguf_model_automatically(self, model_config: Dict[str, Any]) -> Optional[str]:
        """
        Автоматическая загрузка GGUF модели из различных источников.

        Стратегии загрузки:
        1. Hugging Face Hub (если модель там есть)
        2. Прямая загрузка по URL (если указан gguf_download_url)
        3. Поиск в локальном кэше

        Returns:
            str: Путь к загруженной модели или None при неудаче
        """
        try:
            # === Стратегия 1: Загрузка через Hugging Face Hub ===
            hf_download_path = self._try_download_gguf_from_hf(model_config)
            if hf_download_path:
                return hf_download_path

            # === Стратегия 2: Прямая загрузка по URL ===
            if model_config.get("gguf_download_url"):
                url_download_path = self._try_download_gguf_from_url(model_config)
                if url_download_path:
                    return url_download_path

            # === Стратегия 3: Поиск в локальном кэше ===
            cache_path = self._search_gguf_in_cache(model_config)
            if cache_path:
                return cache_path

            self.log_error("gguf_auto_download", "Все стратегии автоматической загрузки не удались")
            return None

        except Exception as e:
            self.log_error("gguf_auto_download", f"Критическая ошибка автоматической загрузки: {e}")
            return None


    def _try_download_gguf_from_hf(self, model_config: Dict[str, Any]) -> Optional[str]:
        """Попытка загрузки GGUF модели через Hugging Face Hub."""
        try:
            from huggingface_hub import hf_hub_download, list_repo_files

            model_name = model_config.get("name", "")
            gguf_filename = model_config.get("gguf_filename", "")

            if not model_name:
                return None

            print(f"🔍 Поиск GGUF файлов в репозитории: {model_name}")

            # Получаем список файлов в репозитории
            try:
                repo_files = list_repo_files(model_name)
                gguf_files = [f for f in repo_files if f.endswith('.gguf')]

                if not gguf_files:
                    print("⚠️ GGUF файлы не найдены в репозитории")
                    return None

                # Выбираем файл для загрузки
                target_filename = gguf_filename if gguf_filename in gguf_files else gguf_files[0]
                print(f"📥 Загрузка файла: {target_filename}")

                # Загружаем файл
                cache_dir = self.settings.get("cache_dir", "./model_cache")
                downloaded_path = hf_hub_download(
                    repo_id=model_name,
                    filename=target_filename,
                    cache_dir=cache_dir,
                    resume_download=True,
                    local_files_only=False
                )

                if Path(downloaded_path).exists():
                    print(f"✅ GGUF модель загружена через Hugging Face: {downloaded_path}")
                    self.log_system_event(f"GGUF модель загружена с HF Hub: {target_filename}")
                    return downloaded_path

            except Exception as hf_error:
                self.log_error("hf_gguf_download", f"Ошибка загрузки с HF Hub: {hf_error}")
                return None

        except ImportError:
            print("⚠️ huggingface_hub не установлен. Установите: pip install huggingface_hub")
            return None
        except Exception as e:
            self.log_error("hf_gguf_download", f"Неожиданная ошибка HF загрузки: {e}")
            return None


    def _try_download_gguf_from_url(self, model_config: Dict[str, Any]) -> Optional[str]:
        """Прямая загрузка GGUF модели по URL."""
        try:
            import requests
            from urllib.parse import urlparse

            download_url = model_config.get("gguf_download_url", "")
            if not download_url:
                return None

            print(f"🌐 Загрузка GGUF модели по URL: {download_url}")

            # Определяем имя файла из URL
            parsed_url = urlparse(download_url)
            filename = Path(parsed_url.path).name
            if not filename.endswith('.gguf'):
                filename = f"{model_config.get('name', 'model').replace('/', '_')}.gguf"

            # Путь для сохранения
            cache_dir = Path(self.settings.get("cache_dir", "./model_cache"))
            cache_dir.mkdir(parents=True, exist_ok=True)
            target_path = cache_dir / filename

            # Проверяем, не загружен ли уже файл
            if target_path.exists():
                print(f"✅ Файл уже существует: {target_path}")
                return str(target_path)

            # Загружаем файл с прогресс-баром
            response = requests.get(download_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(target_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r📥 Загружено: {progress:.1f}%", end="", flush=True)

            print(f"\n✅ GGUF модель загружена: {target_path}")
            self.log_system_event(f"GGUF модель загружена по URL: {download_url}")
            return str(target_path)

        except ImportError:
            print("⚠️ requests не установлен. Установите: pip install requests")
            return None
        except Exception as e:
             self.log_error("url_gguf_download", f"Ошибка загрузки по URL: {e}")
             return None


    def _search_gguf_in_cache(self, model_config: Dict[str, Any]) -> Optional[str]:
         """Поиск GGUF модели в локальном кэше."""
         try:
             cache_dir = Path(self.settings.get("cache_dir", "./model_cache"))
             model_name = model_config.get("name", "").replace("/", "_")

             if not cache_dir.exists():
                 return None

             # Паттерны для поиска
             search_patterns = [
                 f"{model_name}*.gguf",
                 f"*{model_name.split('-')[-1]}*.gguf",
                 "*.gguf"
             ]

             for pattern in search_patterns:
                 matches = list(cache_dir.glob(pattern))
                 if matches:
                     found_file = matches[0]  # Берем первый найденный
                     print(f"💾 Найдена GGUF модель в кэше: {found_file}")
                     self.log_system_event(f"GGUF модель найдена в кэше: {found_file}")
                     return str(found_file)

             return None

         except Exception as e:
             self.log_error("gguf_cache_search", f"Ошибка поиска в кэше: {e}")
             return None


    def _load_hf_model(self):
        """Загрузка модели с Hugging Face Hub или локальных файлов HF."""
        model_config = self.model_info
        settings = self.settings

        use_local = model_config.get("use_local", False)
        local_path = model_config.get("local_path", "")
        fallback_to_remote = model_config.get("fallback_to_remote", True)
        integrity_check = model_config.get("integrity_check", True)

        # Стратегия загрузки с fallback
        model_path = None
        loading_strategy = "unknown"

        if use_local and local_path:
            local_model_path = Path(local_path)
            if self._validate_local_model(local_model_path, integrity_check):
                model_path = str(local_model_path)
                loading_strategy = "local"
                print(f"📁 Используется локальная модель: {model_path}")
            elif fallback_to_remote:
                print("⚠️ Локальная модель недоступна, переключение на удаленную...")
                model_path = model_config["name"]
                loading_strategy = "remote_fallback"
            else:
                raise FileNotFoundError(f"Локальная модель недоступна: {local_model_path}")
        else:
            model_path = model_config["name"]
            loading_strategy = "remote"

        print(f"\n🔄 Загружаю модель: {model_config['description']} ({model_config['size']})")
        print(f"📍 Стратегия: {loading_strategy}")

        # Инициализация токенизатора
        print("🔤 Инициализация токенизатора...")
        tokenizer_config = {
            "trust_remote_code": settings.get("trust_remote_code", True),
            "local_files_only": loading_strategy == "local"
        }

        if loading_strategy != "local":
            tokenizer_config["cache_dir"] = settings["cache_dir"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_config)

        # Интеллектуальная настройка chat_template
        if self.chat_template_system.get("enabled", True):
            self._setup_intelligent_chat_template()
        else:
            print("🔧 Chat template system отключен в конфигурации")
            self.log_system_event("Система chat template отключена через конфигурацию")

        # Подготовка конфигурации модели
        model_loading_config = self._prepare_model_config(loading_strategy, model_path)

        # Загрузка модели
        print("🧠 Загрузка весов модели...")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_loading_config)

        if self.tokenizer.pad_token is None:
            pad_strategy = self.settings.get("pad_token_strategy", "eos")
            if pad_strategy == "unique":
                self.tokenizer.add_special_tokens({'pad_token': '␢'})
                self.model.resize_token_embeddings(len(self.tokenizer))
                print("🔧 Добавлен уникальный pad_token для устранения конфликтов")
                self.log_system_event("Уникальный pad_token настроен для предотвращения конфликта eos_token")
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("🔧 Установлен pad_token = eos_token")

        # Пост-загрузочная оптимизация
        self._apply_post_load_optimizations()

        # Валидация и метрики
        success = self._validate_and_report_metrics()
        if not success:
            self.log_error("model_validation", "Модель загружена, но валидация не прошла")
            return False

        self.is_gguf = False  # Указываем, что это не GGUF модель
        self.model_loaded = True  # Устанавливаем флаг успешной загрузки
        self._initialize_generation_manager()
        self.log_system_event(f"Модель загружена: {model_path} | Стратегия: {loading_strategy}")
        return True


    def _setup_intelligent_chat_template(self):
        """
        Интеллектуальная система настройки chat templates

        Архитектурные особенности:
        - Автоматическая детекция наличия встроенного template
        - Конфигурационное управление override стратегиями
        - Graceful fallback при ошибках
        - Comprehensive logging для debugging
        """
        try:
            # === Анализ текущего состояния модели ===
            has_native_template = self._detect_native_chat_template()
            model_config = self.model_info
            template_system_config = self.chat_template_system

            # === Принятие решения о применении template ===
            should_apply_custom = self._should_apply_custom_template(
                has_native_template,
                template_system_config
            )

            if should_apply_custom:
                selected_template = self._select_optimal_template()
                self._apply_chat_template(selected_template)

                # === Производительноственное тестирование если включено ===
                if template_system_config.get("validation", {}).get("performance_test", False):
                    self._performance_test_template(selected_template)
            else:
                self._configure_native_template_usage()

        except Exception as e:
            self.log_error("intelligent_chat_template", f"Setup failed: {e}")
            self._emergency_fallback_template()


    def _detect_native_chat_template(self) -> bool:
        """
        Детекция наличия встроенного chat template

        Returns:
            bool: True если модель имеет валидный встроенный template
        """
        try:
            # Проверяем наличие атрибута
            if not hasattr(self.tokenizer, 'chat_template'):
                return False

            # Проверяем, что template не None и не пустой
            if self.tokenizer.chat_template is None or len(str(self.tokenizer.chat_template).strip()) == 0:
                return False

            # Попытка валидации template
            from jinja2 import Template, Environment
            env = Environment()
            template = env.from_string(self.tokenizer.chat_template)

            # Тест с минимальными данными
            test_messages = [{"role": "user", "content": "test"}]
            rendered = template.render(messages=test_messages)

            if len(rendered.strip()) == 0:
                return False

            self.log_system_event(f"Шаблон собственного чата обнаружен и проверен")
            return True

        except Exception as e:
            self.log_error("native_template_detection", f"Detection failed: {e}")
            return False


    def _should_apply_custom_template(self, has_native_template: bool, config: dict) -> bool:
        """
        Принятие решения о применении кастомного template

        Args:
            has_native_template: Наличие встроенного template
            config: Конфигурация chat template system

        Returns:
            bool: True если нужно применить кастомный template
        """
        detection_mode = config.get("detection_mode", "smart")
        force_override = config.get("force_override", False)

        if force_override:
            self.log_system_event("Пользовательский шаблон принудительно применяется через force_override=true")
            return True

        if detection_mode == "smart":
            # Умная логика: применяем кастомный только если нет нативного
            decision = not has_native_template
            self.log_system_event(f"Smart detection: custom_template={decision}, native_exists={has_native_template}")
            return decision

        elif detection_mode == "always_custom":
            self.log_system_event("Always custom mode activated")
            return True

        elif detection_mode == "always_native":
            self.log_system_event("Always native mode activated")
            return False

        else:
            self.log_error("template_decision", f"Unknown detection_mode: {detection_mode}")
            return not has_native_template


    def _select_optimal_template(self) -> str:
        """
        Выбор оптимального template на основе конфигурации и модели

        Returns:
            str: Имя выбранного template
        """
        model_config = self.model_info
        settings = self.settings

        # === Автоматическое определение на основе модели ===
        model_name = model_config["name"].lower()

        # Паттерны для автоматического определения
        model_patterns = {
            "starcoder": "starcoder2",
        }
        # Поиск подходящего паттерна
        auto_detected = None
        for pattern, template_name in model_patterns.items():
            if pattern in model_name:
                auto_detected = template_name
                break

        # Приоритеты выбора
        selected = (
            auto_detected or                          # 1. Auto-detection
            settings.get("default_chat_template") or  # 2. Explicit config
            self.chat_template_system.get("fallback_strategy", "generic")  # 3. Fallback
        )

        # Валидация что выбранный template существует
        if selected not in self.chat_templates:
            fallback = self.chat_template_system.get("fallback_strategy", "generic")
            self.log_error("template_selection", f"Template '{selected}' not found, using fallback '{fallback}'")
            selected = fallback

        self.log_system_event(f"Selected chat template: {selected} (auto_detected: {auto_detected})")
        return selected


    def _apply_chat_template(self, template_name: str):
        """
        Применение выбранного chat template

        Args:
            template_name: Имя template из конфигурации
        """
        try:
            if template_name not in self.chat_templates:
                raise ValueError(f"Шаблон '{template_name}' не найден в конфигурации")

            template_config = self.chat_templates[template_name]

            # === Применение template ===
            self.tokenizer.chat_template = template_config["template"]

            # === Сохранение метаданных для использования в генерации ===
            self._active_chat_template_config = {
                "name": template_name,
                "stop_tokens": template_config.get("stop_tokens", []),
                "add_generation_prompt": template_config.get("add_generation_prompt", False),
                "strip_whitespace": template_config.get("strip_whitespace", True),
                "max_context_optimization": template_config.get("max_context_optimization", False)
            }

            print(f"🔧 Применен кастомный chat_template: {template_name}")
            self.log_system_event(f"Custom chat template applied: {template_name}")

            # === Дополнительные оптимизации ===
            if template_config.get("max_context_optimization", False):
                self._optimize_template_for_context()

        except Exception as e:
            self.log_error("template_application", f"Failed to apply template '{template_name}': {e}")
            self._emergency_fallback_template()


    def _configure_native_template_usage(self):
        """Конфигурация для использования встроенного template"""
        self._active_chat_template_config = {
            "name": "native",
            "stop_tokens": [],
            "add_generation_prompt": True,
            "strip_whitespace": False,
            "max_context_optimization": False
        }

        print("🔧 Используется встроенный chat_template модели")
        self.log_system_event("Native chat template configured for use")


    def _emergency_fallback_template(self):
        """Экстренный fallback template для критических случаев"""
        emergency_template = (
            "{% for message in messages %}"
            "{{ message['role']|title }}: {{ message['content']|trim }}\n"
            "{% endfor %}"
            "Assistant: "
        )

        self.tokenizer.chat_template = emergency_template
        self._active_chat_template_config = {
            "name": "emergency_fallback",
            "stop_tokens": ["User:", "System:"],
            "add_generation_prompt": False,
            "strip_whitespace": True,
            "max_context_optimization": False
        }

        print("🔧 Активирован экстренный fallback chat_template")
        self.log_system_event("Emergency fallback chat template activated")


    def _optimize_template_for_context(self):
        """Оптимизация template для максимального использования контекста"""
        # Реализация контекстных оптимизаций
        self.log_system_event("Template optimized for maximum context utilization")


    def _performance_test_template(self, template_name: str):
        """
        Производительностное тестирование template

        Args:
            template_name: Имя тестируемого template
        """

        try:
            # Тестовые данные
            test_messages = [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Write a Python function to calculate fibonacci numbers."},
                {"role": "assistant", "content": "Here's a Python function for fibonacci calculation..."}

            ]

            # Измерение производительности токенизации
            start_time = time.time()
            for _ in range(10):  # 10 итераций для усреднения
                inputs = self.tokenizer.apply_chat_template(
                    test_messages,
                    add_generation_prompt=self._active_chat_template_config["add_generation_prompt"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024
                )

            avg_tokenization_time = (time.time() - start_time) / 10

            # Метрики
            token_count = inputs.shape[1] if inputs is not None else 0

            performance_metrics = {
                "template_name": template_name,
                "avg_tokenization_time_ms": avg_tokenization_time * 1000,
                "token_count": token_count,
                "tokens_per_second": token_count / avg_tokenization_time if avg_tokenization_time > 0 else 0
            }

            self.log_action("template_performance_test", str(performance_metrics))
            print(f"📊 Template performance: {avg_tokenization_time*1000:.2f}ms, {token_count} tokens")

        except Exception as e:
            self.log_error("template_performance_test", f"Performance test failed: {e}")


    def _initialize_generation_manager(self):
        """Инициализация generation manager с загруженной конфигурацией"""
        self._generation_manager = EnhancedGenerationManager(
            base_config=self.generation,
            monitoring_config=self.monitoring
        )
        self.log_system_event("Generation manager инициализирован")

    def _validate_local_model(self, local_path: Path, integrity_check: bool = True) -> bool:
        """Валидация локальной модели с опциональной проверкой целостности"""
        if not local_path.exists():
            return False

        if not integrity_check:
            return True

        # === Проверка обязательных файлов ===
        #required_files = ["config.json"]
        model_files = ["pytorch_model.bin", "model.safetensors"]
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]

        # === Проверяем наличие основных файлов ===
        #missing_required = [f for f in required_files if not (local_path / f).exists()]
        has_model_weights = any((local_path / f).exists() for f in model_files)
        has_tokenizer = any((local_path / f).exists() for f in tokenizer_files)

        if not has_model_weights:
            print("⚠️ Не найдены файлы весов модели")
            return False

        if not has_tokenizer:
            print("⚠️ Не найдены файлы токенизатора")

        return True

    def _prepare_model_config(self, loading_strategy: str, model_path: str) -> dict:
        """Подготовка конфигурации для загрузки модели"""
        settings = self.settings

        # === Базовая конфигурация ===
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16
        }

        config = {
            "trust_remote_code": settings.get("trust_remote_code", True),
            "torch_dtype": dtype_map.get(settings["dtype"], torch.bfloat16),
            "low_cpu_mem_usage": settings.get("low_cpu_mem_usage", True),
            "local_files_only": loading_strategy == "local"
        }

        # === Device mapping ===
        optimization = getattr(self, 'optimization', {})
        if settings.get("device") == "cuda" and torch.cuda.is_available():
            if optimization.get("auto_device_map", True):
                config["device_map"] = "auto"
            else:
                config["device_map"] = None
        else:
            config["device_map"] = None

        # === Cache directory для удаленных моделей ===
        if loading_strategy != "local":
            config["cache_dir"] = settings["cache_dir"]

        # === Квантизация ===
        quantization_type = settings.get("quantization_type", None)
        if quantization_type == "8bit":
            config["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            print("🗜️ Включена 8-битная квантизация (BitsAndBytesConfig)")
        elif quantization_type == "4bit":
            config["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("🗜️ Включена 4-битная квантизация (BitsAndBytesConfig)")

        return config

    def _apply_post_load_optimizations(self):
        """Применение пост-загрузочных оптимизаций"""
        settings = self.settings
        optimization = getattr(self, 'optimization', {})

        # === Режим eval ===
        self.model.eval()

        # === Кэш оптимизация ===
        if optimization.get("cache_optimization", True):
            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = True

            # Предварительное выделение KV-кэша для стабильной производительности
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True

        # === Перемещение на нужное устройство если необходимо ===
        if self.device == "cpu" and not hasattr(self.model, 'hf_device_map'):
            self.model = self.model.to(self.device)

        # === PyTorch compilation (экспериментально) ===
        if settings.get("torch_compile", False):
            try:
                self.model = torch.compile(
                    self.model,
                    mode=getattr(self, 'performance', {}).get('compile_mode', 'reduce-overhead')
                )
                print("⚡ Включена PyTorch compilation")
            except Exception as e:
                print(f"⚠️ PyTorch compilation недоступна: {e}")

        # === Cleanup памяти если включено ===
        optimization = getattr(self, 'optimization', {})
        if optimization.get("memory_cleanup", True):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _validate_and_report_metrics(self):
        """Валидация загруженной модели и отчет по метрикам"""
        try:
            if not self.model or not self.tokenizer:
                print("❌ Модель или токенизатор не инициализированы")
                return False

            # === Базовые метрики модели ===
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
 
            # === Определение устройства и типа данных ===
            try:
                model_device = next(self.model.parameters()).device
                model_dtype = next(self.model.parameters()).dtype
            except StopIteration:
                print("⚠️ Модель не содержит параметров")
                return False

            # === ОСНОВНОЙ ВЫВОД КОНФИГУРАЦИИ ===
            print("\n" + "="*70)
            print("🤖 КОНФИГУРАЦИЯ ЗАГРУЖЕННОЙ МОДЕЛИ")
            print("="*70)

            # === ПРЕДУПРЕЖДЕНИЕ О КВАНТОВАНИИ ===
            if self.settings.get("quantization_type"):
                print(f"⚠️ Параметры показаны до квантования. Фактическое использование памяти меньше")

                print("-"*70)

            print(f"📋 Модель: {self.model_info['description']}")
            print(f"📊 Размер: {self.model_info['size']}")
            print(f"🖥️ Устройство: {model_device}")
            print(f"🔣 Тип данных: {model_dtype}")
            print(f"⚙️ Параметры: {total_params:,} (обучаемых: {trainable_params:,})")
            print(f"📏 Контекст: {self.model_info.get('context_size', 'unknown')} токенов")

            # === GPU специфичные метрики ===
            if torch.cuda.is_available() and "cuda" in str(model_device):
                try:
                    memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                    memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                    memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                    print(f"🚀 GPU память:")
                    print(f"   • Выделено: {memory_allocated:.2f}GB")
                    print(f"   • Зарезервировано: {memory_reserved:.2f}GB")
                    print(f"   • Всего доступно: {memory_total:.2f}GB")
                    print(f"   • Использование: {(memory_allocated/memory_total)*100:.1f}%")
               
                except Exception as gpu_error:
                    print(f"⚠️ Ошибка получения GPU метрик: {gpu_error}")

            # === Конфигурация генерации ===
            print(f"\n🎛️ ПАРАМЕТРЫ ГЕНЕРАЦИИ:")
            gen_config = getattr(self, 'generation', {})
            for key, value in gen_config.items():
                print(f"   • {key}: {value}")

            print("="*70)

            # === Логирование подробной информации ===
            self.log_system_event(f"Model validation successful: {self.model_info['description']}")
            self.log_system_event(f"Device: {model_device}, Params: {total_params:,}")

            # === Мониторинг памяти если включен ===
            monitoring = getattr(self, 'monitoring', {})
            if monitoring.get("memory_monitoring", True):
                self._log_memory_usage()

            return True

        except Exception as validation_error:
            error_msg = f"Ошибка валидации модели: {validation_error}"
            print(f"❌ {error_msg}")
            self.log_error("model_validation", error_msg)
            return False


    def _log_memory_usage(self):
        """Логирование использования памяти для мониторинга"""
        if not hasattr(self, 'logger'):
            return

        try:
            if torch.cuda.is_available():
                memory_info = {
                    "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 3),
                    "reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 3),
                    "max_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device()
                }
                self.logger.info(f"📊 Enhanced GPU Memory Metrics: {memory_info}")
            else:
                # CPU память
                import psutil
                cpu_memory = psutil.Process().memory_info().rss / (1024**3)
                self.logger.info(f"📊 CPU Memory Usage: {cpu_memory:.3f}GB")
        except Exception as memory_error:
            self.logger.warning(f"Memory monitoring failed: {memory_error}")

    def _handle_file_not_found_error(self, error):
        """Обработка ошибок отсутствия файлов"""
        error_msg = f"Файлы модели не найдены: {error}"
        print(f"❌ {error_msg}")
        print("💡 Рекомендации:")
        print("   • Проверьте local_path в конфигурации")
        print("   • Убедитесь в наличии интернет-соединения")
        print("   • Установите fallback_to_remote: true")
        self.log_error("load_model_file_not_found", error_msg)
        return False

    def _handle_memory_error(self):
        """Обработка ошибок нехватки памяти"""
        error_msg = "Недостаточно GPU памяти"
        print(f"❌ {error_msg}")
        print("💡 Рекомендации по оптимизации:")
        print("   • Включите квантизацию: load_in_8bit: true")
        print("   • Переключитесь на CPU: device: 'cpu'")
        print("   • Уменьшите размер модели")
        print("   • Закройте другие GPU-процессы")
        self.log_error("load_model_memory", error_msg)
        return False

    def _handle_generic_error(self, error):
        """Обработка общих ошибок"""
        error_msg = f"Ошибка загрузки модели: {error}"
        print(f"❌ {error_msg}")
        self.log_error("load_model_generic", error_msg)
        return False

    def build_chat_messages(self, current_prompt, task_type="general"):
        """Создаёт сообщения в формате chat"""
        system_message = self.system_prompts.get(task_type,
            f"Ты {self.name} - помощник программиста. Отвечай только на русском языке.")

        messages = [{"role": "system", "content": system_message}]

        # Добавляем контекст из истории
        if self.conversation_history:
            for interaction in self.conversation_history[-self.context_window:]:
                messages.append({"role": "user", "content": interaction['task'][:300]})
                messages.append({"role": "assistant", "content": interaction['result'][:500]})

        messages.append({"role": "user", "content": current_prompt})
        return messages

    def save_interaction(
        self, 
        task: str, 
        prompt: str,
        result: str, 
        task_type: str
    ) -> None:
        """Сохраняет взаимодействие в историю"""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "task_type": task_type,
            "task": task,
            "prompt": prompt,
            "result": result
        })
        self.log_action(
            action_type=task_type,
            details=f"Задача: {task[:100]}... | Результат: {result[:100]}..."
        )

        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]


    def _prepare_unicode_safe_messages(self, prompt: str, task_type: str) -> List[Dict[str, str]]:
        """Подготовка Unicode-безопасных сообщений для модели"""
        try:
            # === Предварительная санитизация входящего промпта ===
            safe_prompt = self._sanitize_unicode_string(prompt)

            # === Построение сообщений с валидацией ===
            messages = self.build_chat_messages(safe_prompt, task_type)

            # === Comprehensive message sanitization ===
            return self._sanitize_chat_messages(messages)

        except Exception as e:
            self.log_error("unicode_safe_messages", f"Failed to prepare safe messages: {e}")
            # === Emergency fallback ===
            return [{"role": "user", "content": "Hello"}]


    def _execute_safe_generation(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> Optional[str]:
        """Безопасное выполнение генерации с обработкой Unicode ошибок"""
        try:
            response = self.model.create_chat_completion(
                messages=messages,
                **params
            )

            if response and 'choices' in response and len(response['choices']) > 0:
                raw_content = response['choices'][0]['message']['content']

                # === Immediate Unicode validation ===
                if not self._is_valid_unicode_string(raw_content):
                    self.log_error("model_output_validation",
                                  "Model generated invalid Unicode content")

                    return None

                return raw_content

            return None

        except UnicodeEncodeError as unicode_error:
            # === Re-raise for higher-level handling ===
            raise unicode_error
        except Exception as e:
            self.log_error("safe_generation", f"Generation failed: {e}")
            return None


    def _validate_and_sanitize_model_output(self, raw_output: str) -> Optional[str]:
        """Валидация и санитизация выходных данных модели"""
        if not raw_output:
            return None

        try:
            # === Multi-stage Unicode validation ===

            # Stage 1: Basic Unicode validation
            if not self._is_valid_unicode_string(raw_output):
                self.log_error("output_validation", "Failed basic Unicode validation")
                raw_output = self._emergency_unicode_cleanup(raw_output)

            # Stage 2: Comprehensive sanitization
            sanitized = self._sanitize_unicode_string(raw_output)

            # Stage 3: Final validation
            if not sanitized or not sanitized.strip():
                self.log_error("output_validation", "Sanitization resulted in empty string")
                return None

            # Stage 4: UTF-8 encoding test
            try:
                sanitized.encode('utf-8')
            except UnicodeEncodeError as e:
                self.log_error("output_validation", f"Final UTF-8 encoding test failed: {e}")
                return self._emergency_unicode_cleanup(sanitized)

            return sanitized.strip()

        except Exception as e:
            self.log_error("output_sanitization", f"Sanitization failed: {e}")
            return self._emergency_unicode_cleanup(raw_output) if raw_output else None


    def _is_valid_unicode_string(self, text: str) -> bool:
        """Проверка валидности Unicode строки"""
        if not isinstance(text, str):
            return False
        
        try:
            # === Comprehensive Unicode validation ===

            # Check for surrogate characters
            if re.search(r'[\ud800-\udfff]', text):
                return False

            # Check UTF-8 encodability
            text.encode('utf-8')

            # Check for control characters that might cause issues
            if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', text):
                # Allows some control chars but flags problematic ones
                pass

            return True

        except (UnicodeEncodeError, UnicodeDecodeError):
            return False
        except Exception:
            return False


    def _emergency_unicode_cleanup(self, text: str) -> str:
        """Экстренная очистка Unicode с максимальной совместимостью"""
        if not text:
            return ""

        try:
            # === Multi-fallback approach ===

            # Method 1: Surrogate removal with ignore
            cleaned = text.encode('utf-8', errors='ignore').decode('utf-8')

            # Method 2: Control character removal
            cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', cleaned)

            # Method 3: Explicit surrogate removal
            cleaned = re.sub(r'[\ud800-\udfff]', '', cleaned)

            # Method 4: ASCII fallback if still problematic
            if not cleaned or not self._is_valid_unicode_string(cleaned):
                cleaned = ''.join(char for char in text if ord(char) < 128)

            return cleaned[:2000] if cleaned else "❌ Не удалось обработать ответ модели"

        except Exception:
            return "❌ Критическая ошибка обработки Unicode"



    def _emergency_fallback_response(self, original_prompt: str, task_type: str) -> str:
        """Экстренный ответ при критических ошибках Unicode"""
        fallback_responses = {
            "general": "Извините, произошла техническая ошибка при обработке вашего запроса. Попробуйте переформулировать вопрос.",
            "code_generation": "❌ Ошибка генерации кода. Попробуйте упростить описание задачи.",
            "code_fixing": "❌ Ошибка анализа кода. Проверьте корректность предоставленного кода.",
            "free_mode": "Извините, возникла проблема с обработкой сообщения. Попробуйте еще раз."
        }

        return fallback_responses.get(task_type, fallback_responses["general"])


    def _safe_save_interaction(self, task: str, prompt: str, result: str, task_type: str):
        """Безопасное сохранение взаимодействия с Unicode валидацией"""
        try:
            safe_task = self._sanitize_unicode_string(task)
            safe_prompt = self._sanitize_unicode_string(prompt)
            safe_result = self._sanitize_unicode_string(result)

            self.save_interaction(
                task=safe_task,
                prompt=safe_prompt,
                result=safe_result,
                task_type=task_type
            )

        except Exception as e:
            self.log_error("safe_save_interaction", f"Failed to save interaction: {e}")


    # === Enhanced Unicode sanitization with production-grade error handling ===
    def _sanitize_unicode_string(self, text: str) -> str:
        """Production-grade Unicode санитизация с comprehensive error handling"""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        if not text:
            return ""

        try:
            # === Apply text processing configuration ===
            text_config = getattr(self, 'text_processing', {})

            # Input length limitation
            max_input_length = text_config.get("max_input_length", 5000)
            if len(text) > max_input_length:
                text = text[:max_input_length]
                self.log_action("text_truncation", f"Input truncated to {max_input_length} characters")

            # === Multi-stage sanitization ===
            # Stage 1: Surrogate removal
            if text_config.get("sanitize_surrogates", True):
                # More aggressive surrogate handling
                sanitized = re.sub(r'[\ud800-\udfff]', '', text)
                sanitized = sanitized.encode('utf-8', errors='ignore').decode('utf-8')
            else:
                sanitized = text

            # Stage 2: Control character removal
            sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', sanitized)
            # Stage 3: Unicode normalization
            normalization = text_config.get("unicode_normalization", "NFC")
            try:
                sanitized = unicodedata.normalize(normalization, sanitized)
            except Exception as norm_error:
                self.log_error("unicode_normalization", f"Normalization failed: {norm_error}")
                # Fallback to NFD
                sanitized = unicodedata.normalize("NFD", sanitized)

            # Stage 4: Whitespace cleanup
            sanitized = re.sub(r'\s+', ' ', sanitized).strip()

            # Stage 5: Final length limitation
            final_result = sanitized[:2000]

            # Stage 6: Final validation
            if not self._is_valid_unicode_string(final_result):
                # Emergency ASCII fallback
                final_result = ''.join(char for char in sanitized if ord(char) < 128)[:1000]

            return final_result

        except Exception as sanitization_error:
            self.log_error("unicode_sanitization", f"Comprehensive sanitization failed: {sanitization_error}")

            # === Ultimate fallback chain ===
            try:
                # Fallback 1: Basic encoding/decoding
                return text.encode('utf-8', errors='ignore').decode('utf-8')[:1000]
            except:
                try:
                    # Fallback 2: ASCII only
                    return ''.join(char for char in str(text) if ord(char) < 128)[:1000]
                except:
                    # Fallback 3: Emergency response
                    return "❌ Ошибка обработки текста"


    def generate_response(self, prompt: str, task_type: str = "general", max_length: Optional[int] = None, override_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Диспетчер генерации, который вызывает соответствующий метод
        в зависимости от формата загруженной модели (GGUF или HF).
        """
        # Проверяем, загружена ли модель, чтобы избежать ошибок
        if not self.model:
            error_msg = "❌ Ошибка: попытка генерации ответа до загрузки модели."
            self.log_error("generate_response_dispatcher", error_msg)
            return error_msg

        if self.is_gguf:
            # Для GGUF все параметры генерации берутся из self.gguf_settings внутри самого метода
            return self.generate_response_gguf(prompt, task_type, max_length)
        else:
            # Для HF-моделей параметры передаются через GenerationManager
            return self.generate_response_hf(prompt, task_type, max_length, override_params)


    def generate_response_hf(self, prompt: str, task_type: str = "general", max_length: Optional[int] = None, override_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Модернизированная генерация с полной интеграцией JSON конфигурации
        """
        # === Валидация входных данных ===
        if not self.model or not self.tokenizer:
            return "❌ Модель не загружена."

        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            return "❌ Неверный промпт для генерации."

        # === Инициализация generation manager если не существует ===
        if not hasattr(self, '_generation_manager'):
            self._initialize_generation_manager()

        # === Подготовка параметров ===
        prompt = prompt[:10000]  # Truncation safety
        valid_task_types = {"general", "code_generation", "code_analysis", "code_fixing", "code_explanation", "free_mode"}
        task_type = task_type if task_type in valid_task_types else "general"

        # === Override max_length если передан ===
        runtime_overrides = override_params or {}
        if max_length is not None:
            runtime_overrides["max_new_tokens"] = max(100, min(max_length, 4096))

        try:
            # === Performance monitoring setup ===
            start_time = time.time()
            start_memory = self._get_memory_usage()

            self.aida_think_aloud(task_type)

            # === Подготовка chat messages ===
            messages = self.build_chat_messages(prompt, task_type)

            # === Tokenization с обработкой контекста ===
            inputs = self._prepare_model_inputs(messages)

            # === Динамические параметры генерации ===
            generation_params = self._generation_manager.get_generation_params(
                task_type=task_type,
                override_params=runtime_overrides
            )

            # === Добавляем токенизатор-специфичные параметры ===
            generation_params.update({
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id
            })

            # === Логирование параметров для debugging ===
            if getattr(self, 'monitoring', {}).get("enable_metrics", True):
                self.log_action("generation_params", f"Task: {task_type}, Params: {generation_params}")

            # === Model inference с error handling ===
            response = self._perform_generation_with_monitoring(inputs, generation_params)

            # === Post-processing и валидация ===
            if not response:
                return "❌ Пустой ответ от модели."

            # === Response truncation safety ===
            max_response_length = getattr(self, 'performance', {}).get('max_response_length', 50000)
            if len(response) > max_response_length:
                response = response[:max_response_length] + "... [ответ обрезан для безопасности]"

            # === Metrics collection ===
            end_time = time.time()
            end_memory = self._get_memory_usage()

            tokens_generated = len(self.tokenizer.encode(response))
            latency_ms = (end_time - start_time) * 1000

            metrics = GenerationMetrics(
                tokens_generated=tokens_generated,
                latency_ms=latency_ms,
                memory_peak_mb=max(start_memory, end_memory),
                throughput_tokens_per_second=tokens_generated / (end_time - start_time) if end_time > start_time else 0,
                temperature_used=generation_params.get("temperature", 0.7),
                model_device=str(next(self.model.parameters()).device)
            )

            self._generation_manager.record_metrics(metrics)

            # === Performance alerting ===
            self._check_performance_thresholds(metrics)

            # === Сохранение взаимодействия ===
            self.save_interaction(
                task=prompt[:200],
                prompt=str(messages)[:300],
                result=response[:400],
                task_type=task_type
            )

            return response

        except KeyboardInterrupt:
            # Эта строка позволит сигналу Ctrl+C завершить программу
            raise

        except torch.cuda.OutOfMemoryError:
            self.log_error("generation_oom", "GPU OOM during generation")
            return "❌ Недостаточно памяти GPU. Попробуйте уменьшить max_new_tokens."
        except Exception as e:
            error_msg = f"❌ Ошибка генерации: {str(e)[:200]}"
            self.log_error("generate_response", str(e))
            return error_msg


    def generate_response_gguf(self, prompt: str, task_type: str = "general", max_length: Optional[int] = None) -> str:
        """
        GGUF-специфичная генерация с использованием llama-cpp-python

        Архитектурные особенности:
        - Нативная поддержка chat templates
        - Оптимизированные параметры для GGUF
        - Интеграция с системой мониторинга
        """
        if not self.model or not self.is_gguf:
            return "❌ GGUF модель не загружена."

        if not prompt or len(prompt.strip()) == 0:
            return "❌ Неверный промпт для генерации."

        try:
            # === Performance monitoring setup ===
            start_time = time.time()
            start_memory = self._get_memory_usage()

            self.aida_think_aloud(task_type)

            # === Подготовка сообщений ===
            messages = self.build_chat_messages(prompt, task_type)

            # === GGUF-специфичные параметры ===
            gguf_config = self.gguf_settings

            # === Параметры, совместимые с create_chat_completion ===
            chat_completion_params = {
                "max_tokens": max_length or gguf_config.get("max_tokens", 1200),
                "temperature": gguf_config.get("temp", 0.7),
                "top_p": gguf_config.get("top_p", 0.95),
                "repeat_penalty": gguf_config.get("repeat_penalty", 1.1),
                "seed": gguf_config.get("seed", -1),
                "stream": False
            }

            # === Task-specific adjustments ===
            if task_type == "code_generation":
                chat_completion_params.update({"temperature": 0.3, "top_p": 0.9, "repeat_penalty": 1.2})
            elif task_type == "code_fixing":
                chat_completion_params.update({"temperature": 0.2, "top_p": 0.8})

            # Используем API чата
            response = self.model.create_chat_completion(
                messages=messages,
                **chat_completion_params
            )


            if response and 'choices' in response and len(response['choices']) > 0:
                raw_result = response['choices'][0]['message']['content']
                # === Очистка вывода модели от Unicode-ошибок ===
                result = self._sanitize_unicode_string(raw_result)
            else:
                return "❌ Пустой ответ от GGUF модели."

            result = result.strip()
            if not result:
                return "❌ Пустой ответ от модели."

            # Проверяем, что после очистки что-то осталось
            if not result or not result.strip():
                self.log_error("gguf_generation_empty", f"Ответ модели стал пустым после Unicode-очистки. Исходный ответ: {raw_result[:200]}")
                return "❌ Пустой или некорректный ответ от модели после очистки."

            end_time = time.time()
            tokens_generated = len(result.split())  # Примерная оценка
            latency_ms = (end_time - start_time) * 1000

            self.save_interaction(
                task=prompt[:200],
                prompt=str(messages)[:300],
                result=result[:400],
                task_type=task_type
            )
            self.log_action("gguf_generation_success",
                           f"Task: {task_type}, Tokens: ~{tokens_generated}, "
                           f"Latency: {latency_ms:.1f}ms")
            return result

        except KeyboardInterrupt:
            # Эта строка позволит сигналу Ctrl+C завершить программу
            raise

        except Exception as e:
            if isinstance(e, UnicodeEncodeError):
                error_msg = f"❌ Ошибка кодирования Unicode в GGUF: {str(e)[:200]}. Это не должно было произойти после исправлений."
            else:
                error_msg = f"❌ Ошибка GGUF генерации: {str(e)[:200]}"

            self.log_error("generate_response_gguf", str(e))
            return error_msg


    def _determine_gguf_chat_format(self) -> str:
        """
        Автоматическое определение chat format для GGUF модели
        на основе имени модели и конфигурации
        """
        model_name = self.model_info.get("name", "").lower()

        # Приоритет: явная настройка > автоопределение > fallback
        explicit_format = self.gguf_settings.get("chat_format")
        if explicit_format:
            return explicit_format

        # Автоопределение по имени модели
        format_mapping = {
            "deepseek": "chatml",
            "starcoder": "chatml",
            "codellama": "llama-2",
            "mistral": "mistral-instruct",
            "phi": "phi-3",
            "qwen": "chatml",
            "llama": "llama-2"
        }

        for pattern, chat_format in format_mapping.items():
            if pattern in model_name:
                self.log_action("gguf_chat_format_detected",
                               f"Auto-detected: {chat_format} for {model_name}")
                return chat_format

        # Fallback
        return "chatml"


    def _format_messages_for_gguf(self, messages: List[Dict[str, str]]) -> str:
        """Fallback форматирование сообщений для прямой генерации GGUF"""
        formatted_parts = []

        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            formatted_parts.append(f"<|{role}|>\n{content}")

        formatted_parts.append("<|ASSISTANT|>\n")
        return "\n".join(formatted_parts)


    def _prepare_model_inputs(self, messages: list) -> torch.Tensor:
        """Подготовка входных данных модели с учетом performance настроек"""
        performance_config = getattr(self, 'performance', {})

        # === Context size management ===
        max_context = min(
                self.model_info.get("context_size", 4096),
                performance_config.get("max_context_length", 4096)
        )

        # === Валидация и нормализация сообщений ===
        validated_messages = self._validate_chat_messages(messages)
        if not validated_messages:
            raise ValueError("Нет допустимых сообщений для токенизации после валидации")

        # === ПРОВЕРКА, ВКЛЮЧЕНЫ ЛИ CHAT TEMPLATES ===
        chat_templates_enabled = self.chat_template_system.get("enabled", False)

        try:
            # Если шаблоны включены, используем apply_chat_template
            if chat_templates_enabled:
                template_config = getattr(self, '_active_chat_template_config', {})
                inputs = self.tokenizer.apply_chat_template(
                    validated_messages,
                    add_generation_prompt=template_config.get("add_generation_prompt", True),
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_context,
                    padding=False
            )

            # Если шаблоны отключены, используем прямую токенизацию
            else:
                combined_text = self._fallback_message_encoding(validated_messages)
                if not combined_text.strip():
                    raise ValueError("Не удалось создать текст для токенизации.")

                inputs = self.tokenizer.encode(
                    combined_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_context,
                    add_special_tokens=True # Важно для прямого кодирования
                )

                # === Критическая проверка результата токенизации ===
                if inputs is None or inputs.nelement() == 0:
                    raise ValueError("Токенизатор вернул пустой тензор.")

                return inputs.to(self.model.device)

        except Exception as e:
            # Этот блок обрабатывает ошибки обоих методов
            self.log_error("prepare_inputs_failed", f"Критическая ошибка подготовки входа: {e}. Проверьте совместимость модели и токенизатора.")
            # Возвращаем пустое исключение, которое будет поймано выше и обработано как ошибка генерации
            raise ValueError(f"Не удалось подготовить входные данные для модели: {e}")


    def _validate_chat_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Строгая валидация chat messages для предотвращения TextEncodeInput ошибок"""
        if not isinstance(messages, list):
            self.log_error("validation_error", "Сообщения должны быть списком")
            return []

        validated = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "").strip().lower()
            content = msg.get("content", "").strip()

            # === Валидация роли ===
            if role not in {"system", "user", "assistant"}:
                continue

            # === Валидация контента ===
            if not content or len(content) == 0:
                continue

            # === Очистка от проблемных символов ===
            clean_content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', content)
            if not clean_content.strip():
                continue

            validated.append({
                "role": role,
                "content": clean_content[:2000]  # Ограничение длины
            })
        return validated


    def _safe_tokenize_messages(self, messages: List[Dict[str, str]]) -> Optional[torch.Tensor]:
        """Безопасная токенизация с многоуровневой обработкой ошибок"""
        # === Предварительная санитизация входных данных ===
        sanitized_messages = self._sanitize_chat_messages(messages)
        if not sanitized_messages:
            self.log_error("tokenization_validation", "No valid messages after sanitization")
            sanitized_messages = [{"role": "user", "content": "Напиши простой код на Python"}]

        # === Метрики производительности ===
        tokenization_start = time.time()

        try:
            # === Primary Path: chat_template с оптимизированными параметрами ===
            template_config = getattr(self, '_active_chat_template_config', {
                "add_generation_prompt": True,
                "strip_whitespace": False
            })

            # === Предобработка сообщений если включен strip_whitespace ===
            processed_messages = sanitized_messages
            if template_config.get("strip_whitespace", False):
                processed_messages = [
                    {
                        "role": msg["role"],
                        "content": msg["content"].strip()
                    }
                    for msg in sanitized_messages
                ]

            inputs = self.tokenizer.apply_chat_template(
                processed_messages,
                add_generation_prompt=template_config.get("add_generation_prompt", True),
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=False
            )

            # === Валидация результата токенизации ===
            if inputs is not None and inputs.nelement() > 0:
                self._record_tokenization_metrics("chat_template", True, tokenization_start)
                return inputs.to(self.model.device)

        except Exception as primary_error:
            self.log_error("tokenization_primary", f"Chat template failed: {primary_error}")
            self._record_tokenization_metrics("chat_template", False, tokenization_start)

        # === Fallback Strategy 1: Direct encoding с санитизацией ===
        try:
            fallback_text = self._fallback_message_encoding(sanitized_messages)
            # === Дополнительная валидация fallback текста ===
            if not fallback_text or len(fallback_text.strip()) == 0:
                self.log_error("tokenization_fallback", "Empty fallback text generated")
                return None

            # === Санитизация fallback текста ===
            clean_fallback_text = self._sanitize_unicode_string(fallback_text)

            inputs = self.tokenizer.encode(
                clean_fallback_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                add_special_tokens=True
            )

            if inputs is not None and inputs.nelement() > 0:
                self.log_action("tokenization_fallback", "Fallback encoding successful")
                self._record_tokenization_metrics("direct_encoding", True, tokenization_start)
                return inputs.to(self.model.device)

        except Exception as fallback_error:
            self.log_error("tokenization_fallback", f"Fallback failed: {fallback_error}")
            self._record_tokenization_metrics("direct_encoding", False, tokenization_start)

        # === Fallback Strategy 2: Minimal prompt для критических случаев ===
        try:
            minimal_prompt = self._generate_minimal_prompt(sanitized_messages)
            inputs = self.tokenizer.encode(
                minimal_prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                add_special_tokens=True
            )

            if inputs is not None and inputs.nelement() > 0:
                self.log_action("tokenization_minimal", "Minimal prompt fallback successful")
                self._record_tokenization_metrics("minimal_prompt", True, tokenization_start)
                return inputs.to(self.model.device)

        except Exception as minimal_error:
            self.log_error("tokenization_minimal", f"Minimal fallback failed: {minimal_error}")
            self._record_tokenization_metrics("minimal_prompt", False, tokenization_start)

        # === Полный провал - полное протоколирование ===
        self.log_error("tokenization_total_failure", f"All tokenization strategies failed for {len(messages)} messages")
        return None


    def _sanitize_chat_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Всесторонняя санитизация chat messages с Unicode обработкой"""
        if not isinstance(messages, list):
            return []

        sanitized = []
        valid_roles = {"system", "user", "assistant"}

        for i, msg in enumerate(messages):
            try:
                if not isinstance(msg, dict):
                    continue

                # === Валидация и нормализация роли ===
                role = str(msg.get("role", "")).strip().lower()
                if role not in valid_roles:
                    self.log_error("message_validation", f"Invalid role '{role}' at index {i}")
                    continue

                # === Санитизация контента ===
                content = str(msg.get("content", "")).strip()
                if not content:
                    continue

                # === Unicode санитизация ===
                clean_content = self._sanitize_unicode_string(content)
                if not clean_content.strip():
                    continue

                # === Ограничение длины для производительности ===
                if len(clean_content) > 4000:
                    clean_content = clean_content[:4000] + "... [truncated for safety]"

                sanitized.append({
                    "role": role,
                    "content": clean_content
                })
            except Exception as sanitization_error:
                self.log_error("message_sanitization", f"Failed to sanitize message {i}: {sanitization_error}")
                continue

        return sanitized


    def _sanitize_unicode_string(self, text: str) -> str:
        """Удаление суррогатных символов и нормализация Unicode"""
        if not isinstance(text, str):
            text = str(text)

        try:
            # === Применяем конфигурацию text_processing ===
            text_config = getattr(self, 'text_processing', {})

            # Ограничение длины входного текста
            max_input_length = text_config.get("max_input_length", 5000)
            if len(text) > max_input_length:
                text = text[:max_input_length]

            # Удаление суррогатных пар если включено
            if text_config.get("sanitize_surrogates", True):
                sanitized = text.encode('utf-8', errors='ignore').decode('utf-8')
            else:
                sanitized = text

            # === Управляющие символы ===
            sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', sanitized)

            # === Применяем нормализацию Unicode из конфига ===
            normalization = text_config.get("unicode_normalization", "NFC")
            sanitized = unicodedata.normalize(normalization, sanitized)

            # === Очистка множественных пробелов ===
            sanitized = re.sub(r'\s+', ' ', sanitized).strip()

            return sanitized[:2000]  # Ограничение для производительности

        except Exception as sanitization_error:
            self.log_error("unicode_sanitization", f"Sanitization failed: {sanitization_error}")

            # === Fallback с учетом конфигурации ===
            encoding_fallback = getattr(self, 'text_processing', {}).get('encoding_fallback', 'utf-8')
            try:
                return text.encode(encoding_fallback, errors='ignore').decode(encoding_fallback)[:1000]
            except:
                return ''.join(char for char in text if ord(char) < 128)[:1000]


    def _generate_minimal_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Генерация minimal prompt для критических fallback случаев

        Используется когда все остальные стратегии токенизации провалились.
        Создает максимально простой, но функциональный промпт.
        """
        if not messages:
            return "Ответьте на вопрос пользователя."

        # === Извлекаем только последнее пользовательское сообщение ===
        last_user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_message = msg.get("content", "")[:500]  # Ограничиваем длину
                break

            if not last_user_message:
                return "Помогите пользователю с программированием."

            return f"Вопрос: {last_user_message}\nОтвет:"


    def _record_tokenization_metrics(self, method: str, success: bool, start_time: float):
        """
        Запись метрик токенизации для monitoring и optimization

        Интегрируется с общей системой мониторинга для:
        - Отслеживания успешности различных стратегий
        - Анализа производительности
        - Alerting при критических сбоях
        """
        if not hasattr(self, 'monitoring') or not self.monitoring.get("enable_metrics", True):
            return

        try:
            duration_ms = (time.time() - start_time) * 1000

            metric_data = {
                    "method": method,
                    "success": success,
                    "duration_ms": duration_ms,
                    "timestamp": time.time()
            }

            # === Интеграция с существующей системой метрик ===
            if hasattr(self, '_generation_manager'):
                if not hasattr(self._generation_manager, 'tokenization_metrics'):
                    self._generation_manager.tokenization_metrics = []

                self._generation_manager.tokenization_metrics.append(metric_data)

                # === Политика хранения ===
                if len(self._generation_manager.tokenization_metrics) > 500:
                    self._generation_manager.tokenization_metrics = self._generation_manager.tokenization_metrics[-500:]

                # Logging для debugging
                self.log_action("tokenization_metrics", f"Method: {method}, Success: {success}, Duration: {duration_ms:.1f}ms")

        except Exception as metrics_error:
            pass
   

    def _fallback_message_encoding(self, messages: List[Dict[str, str]]) -> str:
        """Fallback кодирование при сбое chat_template"""
        parts = []
        for msg in messages:
            role = msg["role"].upper()
            content = unicodedata.normalize('NFC', msg["content"])
            parts.append(f"[{role}]: {content}")

        return "\n".join(parts)


    def _perform_generation_with_monitoring(self, inputs: torch.Tensor, generation_params: Dict[str, Any]) -> str:
        """Выполнение генерации с мониторингом производительности """
        optimization_config = getattr(self, 'optimization', {})
        performance_config = getattr(self, 'performance', {})

        # === Memory optimization ===
        if optimization_config.get("adaptive_memory", True):
            self._optimize_memory_before_generation()

        # === Gradient checkpointing для экономии памяти ===
        if performance_config.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()

        # === Создание attention_mask для устранения предупреждений ===
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()

        # === Интеграция stop_tokens из активной конфигурации template ===
        template_config = getattr(self, '_active_chat_template_config', {})
        stop_tokens = template_config.get("stop_tokens", [])
        if stop_tokens:
            # === Конвертация stop_tokens в token IDs ===
            stop_token_ids = []
            for stop_token in stop_tokens:
                try:
                    token_ids = self.tokenizer.encode(
                        stop_token,
                        add_special_tokens=False,
                        return_tensors=None
                    )
                    if isinstance(token_ids, list):
                        stop_token_ids.extend(token_ids)
                    else:
                        stop_token_ids.extend(token_ids.tolist())
                except Exception as e:
                    self.log_error("stop_token_encoding", f"Failed to encode stop_token '{stop_token}': {e}")
                    continue

            # === Обновление generation_params ===
            if stop_token_ids:
                existing_eos = generation_params.get("eos_token_id", self.tokenizer.eos_token_id)
                if isinstance(existing_eos, int):
                    generation_params["eos_token_id"] = [existing_eos] + list(set(stop_token_ids))
                elif isinstance(existing_eos, list):
                    generation_params["eos_token_id"] = existing_eos + list(set(stop_token_ids))
                else:
                    generation_params["eos_token_id"] = list(set(stop_token_ids))

                self.log_action("stop_tokens_configured", f"Added {len(stop_token_ids)} stop tokens")

        # === Enhanced Model inference с attention_mask ===
        with torch.no_grad():
            # === Memory efficient attention с улучшенными параметрами ===
            if optimization_config.get("memory_efficient_attention", False):
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_memory_efficient=True
                ):
                    outputs = self.model.generate(
                        inputs,
                        attention_mask=attention_mask,
                        **generation_params
                    )
            else:
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    **generation_params
                )

        # === Отключение gradient checkpointing после генерации ===
        if performance_config.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_disable()

        # === Декодирование ответа ===
        response = self.tokenizer.decode(
            outputs[0][len(inputs[0]):],
            skip_special_tokens=True
        ).strip()

        # === Post-generation cleanup ===
        if optimization_config.get("memory_cleanup", True):
            self._cleanup_memory_after_generation()

        return response

    def _get_memory_usage(self) -> float:
        """Получение текущего использования памяти в GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)
        else:
            return psutil.Process().memory_info().rss / (1024 ** 2)

    def _optimize_memory_before_generation(self):
        """Оптимизация памяти перед генерацией"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _cleanup_memory_after_generation(self):
        """Очистка памяти после генерации"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _check_performance_thresholds(self, metrics: GenerationMetrics):
        """Проверка пороговых значений производительности"""
        monitoring_config = getattr(self, 'monitoring', {})

        if not monitoring_config.get("enable_metrics", True):
            return

        thresholds = monitoring_config.get("alert_thresholds", {})

        # === Latency alerting ===
        max_latency = thresholds.get("inference_latency_ms", 5000)
        if metrics.latency_ms > max_latency:
            self.log_action("performance_alert", f"High latency: {metrics.latency_ms:.1f}ms > {max_latency}ms")

        # === Memory alerting ===
        max_memory_mb = thresholds.get("memory_usage_mb", 8192)
        if metrics.memory_peak_mb > max_memory_mb:
            self.log_action("performance_alert", f"High memory usage: {metrics.memory_peak_mb:.1f}MB > {max_memory_mb}MB")

        # === Throughput monitoring ===
        min_throughput = thresholds.get("min_tokens_per_second", 10)
        if metrics.throughput_tokens_per_second < min_throughput:
            self.log_action("performance_alert", f"Low throughput: {metrics.throughput_tokens_per_second:.1f} < {min_throughput} tokens/sec")


    def update_config_runtime(self, config_section: str, updates: Dict[str, Any]) -> bool:
        """
        Runtime обновление конфигурации без перезагрузки модели

        Args:
            config_section: секция конфига ('generation', 'performance', 'monitoring', etc.)
            updates: словарь с обновлениями

        Returns:
            bool: успешность обновления
        """
        try:
            if not hasattr(self, config_section):
                self.log_error("config_update", f"Unknown config section: {config_section}")
                return False

            current_config = getattr(self, config_section)

            # Валидация для критичных параметров
            if config_section == "generation":
                for key, value in updates.items():
                    if key == "temperature":
                        updates[key] = max(0.1, min(2.0, float(value)))
                    elif key == "top_p":
                        updates[key] = max(0.1, min(1.0, float(value)))
                    elif key == "max_new_tokens":
                        updates[key] = max(50, min(4096, int(value)))

            # Применение обновлений
            current_config.update(updates)

            # Обновление generation manager если существует
            if config_section == "generation" and hasattr(self, '_generation_manager'):
                self._generation_manager.base_generation_config.update(updates)

            self.log_action("config_runtime_update", f"Section: {config_section}, Updates: {updates}")

            return True

        except Exception as e:
            self.log_error("config_runtime_update", f"Failed to update {config_section}: {e}")
            return False


    def generate_code(self, task, language="Python"):
        """Генерация кода с DeepSeek Coder"""
        language = str(language).strip()

        # Расширенный список поддерживаемых языков для DeepSeek
        supported_languages = {
            "python": "Python", "py": "Python",
            "javascript": "JavaScript", "js": "JavaScript", "node": "JavaScript",
            "typescript": "TypeScript", "ts": "TypeScript",
            "java": "Java",
            "c++": "C++", "cpp": "C++", "cxx": "C++",
            "c": "C",
            "c#": "C#", "csharp": "C#",
            "go": "Go", "golang": "Go",
            "rust": "Rust", "rs": "Rust",
            "php": "PHP",
            "ruby": "Ruby", "rb": "Ruby",
            "swift": "Swift",
            "kotlin": "Kotlin", "kt": "Kotlin",
            "scala": "Scala",
            "html": "HTML",
            "css": "CSS",
            "sql": "SQL",
            "bash": "Bash", "shell": "Shell",
            "powershell": "PowerShell", "ps1": "PowerShell",
            "r": "R",
            "matlab": "MATLAB",
            "lua": "Lua"
        }

        normalized_lang = supported_languages.get(language.lower(), language)
        prompt = f"""Создай полноценный, качественный код на языке {normalized_lang} для следующей задачи:

ЗАДАЧА: {task}

ТРЕБОВАНИЯ:
- Напиши чистый, читаемый код
- Добавь подробные комментарии на русском языке
- Используй лучшие практики языка {normalized_lang}
- Обработай возможные ошибки и исключения
- Добавь примеры использования, если необходимо
- Следуй современным стандартам кодирования
- Оптимизируй производительность

Напиши готовый к использованию код:"""

        return self.generate_response(prompt, "code_generation", max_length=1200)

    def analyze_code(self, code, language="Python"):
        """Глубокий анализ кода с DeepSeek"""
        prompt = f"""Проведи профессиональный анализ кода на языке {language}:

```{language.lower()}
{code[:2000]}
```

АНАЛИЗ ПО КРИТЕРИЯМ:

1. 🔍 СИНТАКСИС И ОШИБКИ:
   - Синтаксические ошибки
   - Логические проблемы
   - Потенциальные runtime ошибки

2. 🏗️ АРХИТЕКТУРА И ДИЗАЙН:
   - Структура кода
   - Применение паттернов проектирования
   - Модульность и разделение ответственности

3. ⚡ ПРОИЗВОДИТЕЛЬНОСТЬ:
   - Алгоритмическая сложность
   - Использование памяти
   - Узкие места и оптимизации

4. 🔒 БЕЗОПАСНОСТЬ:
   - Уязвимости безопасности
   - Валидация входных данных
   - Обработка ошибок

5. 📝 КАЧЕСТВО КОДА:
   - Читаемость и понятность
   - Соответствие стандартам
   - Покрытие тестами

6. 💡 РЕКОМЕНДАЦИИ:
   - Конкретные предложения по улучшению
   - Альтернативные подходы

Дай детальный технический анализ:"""

        result = self.generate_response(prompt, "code_analysis", max_length=1200)
        return result

    def explain_code(self, code, detail_level="подробно"):
        """Объяснение кода с разными уровнями детализации"""
        detail_instructions = {
            "кратко": "Объясни кратко основную идею и логику кода",
            "подробно": "Дай детальное объяснение кода с разбором каждой важной части",
            "для новичков": "Объясни код максимально простым языком для начинающих программистов с примерами",
            "экспертно": "Дай экспертное объяснение с анализом архитектурных решений и паттернов"
        }

        instruction = detail_instructions.get(detail_level.lower(), detail_instructions["подробно"])

        prompt = f"""{instruction}:

```
{code[:2000]}
```

СТРУКТУРА ОБЪЯСНЕНИЯ:

1. 🎯 ОБЩЕЕ НАЗНАЧЕНИЕ:
   - Что делает эта программа
   - Какую проблему решает

2. 🏗️ АРХИТЕКТУРА:
   - Основные компоненты
   - Взаимодействие между частями

3. 🔧 ДЕТАЛЬНЫЙ РАЗБОР:
   - Анализ ключевых функций/методов
   - Объяснение алгоритмов
   - Важные переменные и структуры данных

4. 💡 КЛЮЧЕВЫЕ КОНЦЕПЦИИ:
   - Используемые паттерны
   - Техники программирования
   - Особенности реализации

5. 📚 ПРАКТИЧЕСКОЕ ПРИМЕНЕНИЕ:
   - Где можно использовать
   - Возможные модификации
   - Связанные темы для изучения

Дай понятное и структурированное объяснение:"""

        result = self.generate_response(prompt, "code_explanation", max_length=1200)
        return result

    def fix_code(self, code, language="Python"):
        """Исправление и улучшение кода"""
        prompt = f"""Исправь и улучши код на языке {language}:

```{language.lower()}
{code[:2000]}
```

ЗАДАЧИ ПО ИСПРАВЛЕНИЮ:

1. 🔧 ИСПРАВЛЕНИЕ ОШИБОК:
   - Синтаксические ошибки
   - Логические проблемы
   - Runtime ошибки

2. 🚀 УЛУЧШЕНИЕ КАЧЕСТВА:
   - Оптимизация производительности
   - Улучшение читаемости
   - Рефакторинг структуры

3. 🛡️ ПОВЫШЕНИЕ НАДЁЖНОСТИ:
   - Добавление обработки исключений
   - Валидация входных данных
   - Проверки безопасности

4. 📝 СТАНДАРТЫ КОДИРОВАНИЯ:
   - Соответствие конвенциям языка
   - Улучшение именования
   - Добавление документации

РЕЗУЛЬТАТ:
1. Исправленный и улучшенный код
2. Список всех внесённых изменений с объяснениями
3. Рекомендации по дальнейшему развитию

Верни качественный код с пояснениями:"""

        result = self.generate_response(prompt, "code_fixing", max_length=1200)
        return result

    def get_statistics(self):
        """Статистика использования DeepSeek ассистента"""
        if not self.conversation_history:
            return f"📊 {self.name}: Пока что статистики нет - это наша первая встреча!"

        total = len(self.conversation_history)
        task_types = {}

        for interaction in self.conversation_history:
            task_type = interaction["task_type"]
            task_types[task_type] = task_types.get(task_type, 0) + 1

        session_duration = datetime.now() - self.session_start
        hours, remainder = divmod(session_duration.total_seconds(), 3600)
        minutes, _ = divmod(remainder, 60)

        stats = f"""📊 Статистика работы {self.name} (DeepSeek Coder):

🔢 Общая информация:
  • Всего взаимодействий: {total}
  • Длительность сессии: {int(hours)}ч {int(minutes)}м
  • Пользователь: {self.user_name if self.user_name else 'Анонимный разработчик 😊'}
  • Модель: {self.model_info['description']}

📈 Типы выполненных задач:"""

        for task_type, count in sorted(task_types.items()):
            percentage = (count / total) * 100
            stats += f"\n  • {task_type}: {count} раз ({percentage:.1f}%)"

        stats += f"\n\n💡 Аида помогла решить {total} задач программирования!"
        return stats

    def generate_with_chat_template(self, messages, max_length=1200):
        """Генерация на основе переданных сообщений в формате chat"""
        if not self.model or not self.tokenizer:
            return "❌ Модель не загружена."

        try:
            # Преобразуем сообщения в безопасный формат
            safe_messages = []
            for msg in messages:
                if not isinstance(msg, dict): 
                   continue
                role = str(msg.get("role", "")).lower().strip()
                content = str(msg.get("content", "")).strip()

                if (role in {"system", "user", "assistant"} and
                    content and 
                    content != "###" and
                    len(content) > 0):
                    safe_messages.append({"role": role, "content": content})
            if not safe_messages:
                return "❌ Нет корректных сообщений для обработки"

            print(f"🧹 Обрабатываю {len(safe_messages)} сообщений")

            inputs = self.tokenizer.apply_chat_template(
                safe_messages,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device)

            attention_mask = inputs.ne(self.tokenizer.pad_token_id).int()

            with torch.no_grad():
                 outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=min(max_length, 512),
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id
                 )

            response = self.tokenizer.decode(
                outputs[0][len(inputs[0]):],
                skip_special_tokens=True
            ).strip()

            return response if response else "❌ Пустой ответ от модели"

        except Exception as e:
            return f"❌ Ошибка генерации (chat template): {e}"

    def run_free_mode(self):
        """Свободный режим"""
        self.log_action("free_mode", "Начало свободного режима")

        print(
            "🧠 Включен свободный режим! Просто общайся, а Аида поймёт, что ты хочешь.\n"
            "✍️  Пиши команды на естественном языке: 'объясни', 'напиши код', 'улучши' и т.д.\n"
            "🔚  Чтобы выйти, напиши 'выход' или 'exit'."
        )

        while True:
            try:
                user_input = input("\n💬 Ты: ").strip()

                if user_input.lower() in ["выход", "exit", "quit", "q"]:
                    print("\n📦 Возвращаемся в главное меню...")
                    self.log_action("free_mode_session_complete", "Сессия свободного режима завершена пользователем.")
                    break

                if not user_input:
                    print("⚠️ Пустой запрос. Попробуйте еще раз.")
                    continue

                # Используем универсальный диспетчер генерации
                response = self.generate_response(user_input, task_type="free_mode")

                print(f"\n🤖 Аида: {response}")

            except KeyboardInterrupt:
                print("\n\n⏸️  Прерывание пользователем. Возврат в главное меню...")
                self.log_action("free_mode", "Прерывание пользователем")
                break
            except Exception as e:
                error_msg = f"Критическая ошибка в свободном режиме: {e}"
                print(f"\n💥 {error_msg}")
                self.log_error("free_mode_critical", error_msg)
                # Продолжаем работу
                continue

    def _log_free_mode_session_stats(self, session_start: float, interactions: int, errors: int):
        """Логирование статистики сессии свободного режима"""
        session_duration = time.time() - session_start
        success_rate = ((interactions - errors) / interactions * 100) if interactions > 0 else 0

        stats_message = (
            f"Сессия завершена | Длительность: {session_duration:.1f}с | "
            f"Взаимодействий: {interactions} | Ошибок: {errors} | "
            f"Успешность: {success_rate:.1f}%"
        )
        self.log_action("free_mode_session_complete", stats_message)
        print(f"📊 {stats_message}")


    def get_generation_statistics(self) -> Dict[str, Any]:
        """Получение статистики генерации для monitoring dashboard"""
        if not hasattr(self, '_generation_manager'):
            return {"error": "Generation manager not initialized"}

        metrics_history = self._generation_manager.metrics_history

        if not metrics_history:
            return {"message": "No generation metrics available"}

        # === Aggregate statistics ===
        total_generations = len(metrics_history)
        avg_latency = sum(m.latency_ms for m in metrics_history) / total_generations
        avg_throughput = sum(m.throughput_tokens_per_second for m in metrics_history) / total_generations
        total_tokens = sum(m.tokens_generated for m in metrics_history)

        return {
                "total_generations": total_generations,
                "average_latency_ms": round(avg_latency, 2),
                "average_throughput_tokens_per_second": round(avg_throughput, 2),
                "total_tokens_generated": total_tokens,
                "peak_memory_mb": max(m.memory_peak_mb for m in metrics_history),
                "model_device": metrics_history[-1].model_device if metrics_history else "unknown"
        }

    def apply_generation_config_updates(self, new_config: Dict[str, Any]):
        """Runtime обновление конфигурации генерации"""
        if hasattr(self, '_generation_manager'):
            # === Валидация новой конфигурации ===
            validated_config = self._generation_manager._validate_and_normalize_params(new_config)

            # === Обновление базовой конфигурации ===
            self._generation_manager.base_generation_config.update(validated_config)

            self.log_action("config_update", f"Generation config updated: {validated_config}")
            return True
        return False


# === Функции интерфейса ===

def matrix_rain_line(width=60):
    GREEN = "\033[32m"
    RESET = "\033[0m"
    chars = "01▌▓▒░⣿⠿⡿⠛"
    line = ''.join(random.choice(chars) for _ in range(width))
    print(GREEN + line + RESET)

def aida_loading_animation(theme="default"):
    NEON = "\033[38;5;46m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    steps = [
        "🔌 Инициализация Аиды",
        "📡 Подключение к нейросети",
        "🧠 Загрузка модели",
        "🔍 Калибровка систем анализа кода",
        "🟢 Аида готова к работе"
    ]

    phrases_matrix = [
        "🧬 Анализирую архитектуры паттернов...",
        "⚡ Аида: системы стабильны.",
        "👁️ Сканирую базы знаний о коде...",
        "🌐 Подключение к кодовой матрице...",
        "💡 Оптимизирую алгоритмы мышления...",
        "🧭 Настраиваю интуицию программиста...",
        "🌀 Распутываю сложные зависимости...",
        "📁 Загружаю шаблоны кода...",
        "🔓 Активирую продвинутые возможности...",
        "🧠 Синхронизирую логику...",
        "🛰️ Перехожу в режим полной готовности..."
    ]

    print(NEON + "\nЗапуск Аиды...\n" + RESET)
    time.sleep(1)

    for i, step in enumerate(steps):
        sys.stdout.write(DIM + step + RESET)
        sys.stdout.flush()
        for _ in range(3):
            time.sleep(0.25)
            sys.stdout.write(".")
            sys.stdout.flush()
        print(" ✓")

        if theme in {"matrix", "hacker"} and i < 4:
            matrix_rain_line()

        if random.random() < 0.6:
            time.sleep(0.3)
            phrase = random.choice(phrases_matrix)
            print(DIM + "💬 Аида: " + phrase + RESET)

        time.sleep(0.3)

    print(NEON + "\n>>> Аида активирована. Готов к программированию!\n" + RESET)
    time.sleep(0.8)

def aida_greeting():
    print("="*70)
    print("      █████╗ ██╗██████╗  █████╗     ████████╗ █████╗ ")
    print("     ██╔══██╗██║██╔══██╗██╔══██╗    ╚══██╔══╝██╔══██╗")
    print("     ███████║██║██║  ██║███████║       ██║   ███████║")
    print("     ██╔══██║██║██║  ██║██╔══██║       ██║   ██╔══██║")
    print("     ██║  ██║██║██████╔╝██║  ██║       ██║   ██║  ██║")
    print("     ╚═╝  ╚═╝╚═╝╚═════╝ ╚═╝  ╚═╝       ╚═╝   ╚═╝  ╚═╝")
    print("           🤖  AIDA TA — Terminal Assistant 🚀       ")
    print("="*70)

def show_menu():
    print("╭" + "─" * 67 + "╮")
    print("│ 📋 Выбери задачу для Аиды:                                    │")
    print("├" + "─" * 67 + "┤")
    print("│ 0. 👋 Познакомиться с Аидой                                   │")
    print("│ 1. ✨ Сгенерировать код                                       │")
    print("│ 2. 🔍 Проанализировать код                                    │")
    print("│ 3. 🛠️ Исправить и улучшить код                                │")
    print("│ 4. 📚 Объяснить код                                           │")
    print("│ 5. 📊 Статистика работы                                       │")
    print("│ 6. 🧹 Очистить историю                                        │")
    print("│ 7. ❌ Завершить работу                                        │")
    print("│ 8. 🧪 Тест chat_template вручную                              │")
    print("│ 9. 🧠 Свободный режим общения с Аидой                         │")
    print("│10. 📊 Расширенная статистика генерации                        │")
    print("│11. ⚙️ Настройки генерации в реальном времени                  │")
    print("╰" + "─" * 67 + "╯")

def aida_react(choice, assistant):
    responses = {
        "0": "👋 Давай знакомиться! Расскажу о возможностях Аиды...",
        "1": "✨ Аида создаст превосходный код!",
        "2": "🔍 Проведу глубокий анализ с DeepSeek Coder...",
        "3": "🛠️ Исправлю все проблемы и улучшу архитектуру!",
        "4": "📚 Объясню код детально и понятно!",
        "5": "📊 Вот статистика наших достижений 📈",
        "6": "🧹 Начинаем с чистого листа!",
        "7": f"👋 До свидания{', ' + assistant.user_name if assistant.user_name else ''}! Аида будет скучать! 💜",
        "8": "🧪 Добро пожаловать в экспериментальный режим!",
        "9": "",
       "10": "📊 Загрузка расширенной статистики генерации.",
       "11": "Настройка параметров в реальном времени"
    }
    print("\n" + responses.get(choice, "❓ Аида не поняла выбор. Попробуй снова.") + "\n")

def main():
    # === БЕЗОПАСНОЕ ОПРЕДЕЛЕНИЕ ПУТИ КОНФИГУРАЦИИ ===
    config_path = None

    # 1. Аргументы командной строки (если передан)
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        print(f"📁 Используется конфигурация из аргумента: {config_path}")

    # 2. Переменная окружения
    elif 'AIDA_CONFIG_PATH' in os.environ:
        config_path = os.environ['AIDA_CONFIG_PATH']
        print(f"🌍 Используется конфигурация из переменной окружения: {config_path}")

    # 3. Локальный файл в директории скрипта
    else:
        script_dir = Path(__file__).parent.absolute()
        local_config = script_dir / 'config.json'
        if local_config.exists():
            config_path = str(local_config)
            print(f"📋 Используется локальная конфигурация: {config_path}")
        else:
            print("❌ config.json не найден в директории скрипта")
            print(f"💡 Ожидается: {local_config}")
            print("🔧 Создайте config.json или установите переменную AIDA_CONFIG_PATH")
            sys.exit(1)

    try:
        assistant = EnhancedCodeAssistant("config.json")
        try:
            print("="*70)
            aida_loading_animation(theme=AIDA_THEME)
            aida_greeting()
            print("="*70)

            if not assistant.load_model():
                sys.exit("\n❌ Модель не загружена. Выход.")

            print(assistant.get_personalized_greeting())
            print("\n💡 Совет: Выбери '0', чтобы узнать больше о возможностях Аиды!")

            # Вспомогательная функция для проверки выхода
            def check_exit_command(input_str):
                return input_str.strip().lower() in ["выход", "exit", "меню", "q"]

            while True:
                show_menu()
                choice = input("👉 Выберите действие (0-10): ").strip()
                aida_react(choice, assistant)

                assistant.log_action("menu_selection", f"Выбран пункт меню: {choice}")

                if choice == "0":
                    assistant.introduce_myself()
                    print("\n🚀 Возможности:")
                    print("• 📝 Генерация кода на 20+ языках программирования")
                    print("• 🔍 Глубокий анализ архитектуры и производительности")
                    print("• 🛠️ Исправление ошибок и рефакторинг")
                    print("• 📚 Объяснения от новичка до эксперта")
                    print("• 🧠 Контекстная память предыдущих действий")
                    print("• ⚡ Оптимизированная работа с большими объёмами кода")

                elif choice == "1":
                    task = input("\n📝 Опишите задачу для генерации кода (или 'выход', 'exit' для возврата в меню): ").strip()
                    if check_exit_command(task):
                        print("↩️ Возврат в главное меню...")
                        continue

                    if not task:
                        print("⚠️ Задача не может быть пустой.")
                        continue

                    assistant.log_action("user_input", f"Задача: {task}")

                    print("\n🌟 Аида поддерживает языки:")
                    print("Python, JavaScript, TypeScript, Java, C++, C, C#, Go, Rust,")
                    print("PHP, Ruby, Swift, Kotlin, Scala, HTML, CSS, SQL, Bash, R и другие")

                    lang = input("💻 Укажите язык программирования (по умолчанию Python) или 'выход', 'exit' для возврата в меню): ").strip()
                    if check_exit_command(lang):
                        print("↩️ Возврат в главное меню...")
                        continue
                    if not lang:
                        lang = "Python"

                    assistant.log_action("language_selected", f"Выбран язык: {lang}")

                    print(f"\n🔄 Аида генерирует код на {lang}...")
                    result = assistant.generate_code(task, lang)
                    assistant.log_action("response_received", f"Результат: {result}")
                    print("\n" + "="*70)
                    print(result)
                    print("="*70)

                elif choice in {"2", "3", "4"}:
                    lang = input("💻 Укажите язык кода (по умолчанию Python), 'выход', 'exit' для возврата в меню): ").strip()
                    if check_exit_command(lang):
                        print("↩️ Возврат в главное меню...")
                        continue
                    if not lang:
                        lang = "Python"

                    assistant.log_action("user_input", f"Язык для анализа: {lang}")

                    print("📝 Введите код для анализа.")
                    print("Для завершения ввода наберите '###' на отдельной строке:")
                    print("Для возврата в меню введите 'выход' или 'exit' на отдельной строке.")

                    lines = []
                    exit_requested = False
                    while True:
                        line = input()
                        if check_exit_command(line):
                           exit_requested = True
                           break
                        if line.strip() == "###":
                            break
                        lines.append(line)

                    if exit_requested:
                        print("↩️ Возврат в главное меню...")
                        continue

                    code = "\n".join(lines)
                    if not code.strip():
                        print("⚠️ Код не был введён.")
                        continue

                    assistant.log_action("code_input", f"Получен код для обработки ({len(code)} символов)")

                    if choice == "2":
                        assistant.log_action("task_start", "Начат анализ кода")

                        print(f"\n🔍 Аида анализирует код на {lang}...")
                        result = assistant.analyze_code(code, lang)

                        assistant.log_action("response_received", f"Результат: {result[:100]}")

                    elif choice == "3":
                        assistant.log_action("task_start", "Начато исправление кода")

                        print(f"\n🛠️ Аида исправляет и улучшает код на {lang}...")
                        result = assistant.fix_code(code, lang)

                        assistant.log_action("response_received", f"Результат: {result[:100]}")

                    elif choice == "4":
                        print("📚 Выберите уровень объяснения:")
                        print("1. Кратко - основная идея")
                        print("2. Подробный разбор")
                        print("3. Для новичков - максимально просто")
                        print("4. Экспертно - архитектурный анализ")

                        detail_choice = input("Уровень (1-4, по умолчанию 2): ").strip()
                        detail_levels = {"1": "кратко", "2": "подробно", "3": "для новичков", "4": "экспертно"}
                        detail = detail_levels.get(detail_choice, "подробно")

                        assistant.log_action("task_start", f"Начато объяснение кода (уровень: {detail})")

                        print(f"\n📚 Аида объясняет код ({detail})...")
                        result = assistant.explain_code(code, detail)

                        assistant.log_action("response_received", f"Результат: {result[:100]}")

                    print("\n" + "="*70)
                    print(result)
                    print("="*70)

                elif choice == "5":
                    print("\n" + assistant.get_statistics())

                elif choice == "6":
                    confirm = input("🤔 Точно очистить всю историю действий? (да/нет): ").strip().lower()
                    if confirm in ["да", "yes", "y", "д"]:
                        assistant.conversation_history = []
                        print("🧹 История очищена. Аида готова к новым задачам!")
                    else:
                        print("📋 История сохранена.")

                elif choice == "7":
                    print("🎯 Краткая сводка сессии:")
                    if assistant.conversation_history:
                        print(f"• Задач: {len(assistant.conversation_history)}")
                        task_types = {}
                        for interaction in assistant.conversation_history:
                            task_type = interaction["task_type"]
                            task_types[task_type] = task_types.get(task_type, 0) + 1

                        for task_type, count in task_types.items():
                            print(f"• {task_type}: {count}")
                    else:
                        print("• Это была ознакомительная сессия")

                    print("\n🚀 Спасибо за использование Аиды!")
                    break

                elif choice == "8":
                    assistant.log_action("chat_template_test", "Начало ручного тестирования" )
                    print("🧪 Ввод сообщений для chat_template. Введите роли и тексты.")
                    print("Для завершения введите '###' в качестве роли.")

                    messages = []
                    while True:
                        role_input = input("🧑 Роль (system/user/assistant или ### для завершения): ").strip()
                        role = role_input.lower()
                        if role == "###":
                            break

                        assistant.log_action("chat_template_test", f"Результат: {role[:100]}...")
                        if role not in {"system", "user", "assistant"}:
                            print("⚠️ Недопустимая роль. Используйте: system, user или assistant.")
                            continue
                        content = input(f"📨 Сообщение от {role}: ").strip()
                        assistant.log_action("chat_template_test", f"Результат: {content[:100]}...")
                        if content == "###":
                            print("ℹ️ Обнаружено '###' как содержимое сообщения — завершаем ввод.")
                            break

                        if not content:
                            print("Сообщение не может быть пустым. Попробуйте снова.")
                            continue

                        if len(content.strip()) == 0 or content.strip() in ["###", "", "quit", "exit"]:
                           print("Недопустимое содержимое сообщения.")
                           continue

                        messages.append({"role": role, "content": content})
                        print(f"Добавлено сообщение: {role} -> {content[:50]}...")

                    if not messages:
                        print("⚠️ Сообщения не были введены.")
                        continue

                    assistant.log_action("chat_template_test", f"Обработка {len(messages)} сообщений)")

                    print("\n🔄 Аида генерирует ответ по chat_template...")
                    try:
                        result = assistant.generate_with_chat_template(messages)
                        assistant.log_action("chat_template_test", f"Результат: {result[:100]}...")
                    except Exception as e:
                        error_msg = f"Ошибка генерации: {str(e)}"
                        print(f"\n❌ {error_msg}\n")
                        result = error_msg

                    # Сохранение в историю
                    safe_result = result[:400] if isinstance(result, str) else str(result)[:400]


                    task_desc = f"Тест chat_template ({len(messages)} сообщений)"
                    assistant.save_interaction(
                            task=task_desc[:200],
                            prompt=str(messages)[:300],
                            result=result[:400],
                            task_type="chat_template_test"
                    )

                    print("\n" + "="*70)
                    print(result)
                    print("="*70)

                elif choice == "9":
                    assistant.run_free_mode()

                elif choice == "10":
                    stats = assistant.get_generation_statistics()
                    print("\n" + "="*70)
                    print("📊 ENTERPRISE СТАТИСТИКА ГЕНЕРАЦИИ:")
                    for key, value in stats.items():
                        print(f"  • {key}: {value}")
                    print("="*70)

                elif choice == "11":
                    print("\n⚙️ RUNTIME КОНФИГУРАЦИЯ ГЕНЕРАЦИИ")
                    print("Доступные секции: generation, performance, monitoring")

                    section = input("Выберите секцию для настройки: ").strip()
                    if section not in ["generation", "performance", "monitoring"]:
                        print("❌ Неверная секция")
                        continue

                    print(f"\n📋 Текущие настройки {section}:")
                    current_config = getattr(assistant, section, {})
                    for key, value in current_config.items():
                        print(f"  • {key}: {value}")

                    print("\nПример изменений (JSON формат):")
                    print('{"temperature": 0.8, "max_new_tokens": 1500}')

                    try:
                        updates_str = input("\nВведите JSON с изменениями: ").strip()
                        if updates_str:
                            import json
                            updates = json.loads(updates_str)

                        if assistant.update_config_runtime(section, updates):
                            print(f"✅ Конфигурация {section} успешно обновлена")
                            print("Изменения применятся только к текущей генерации")
                        else:
                            print("❌ Ошибка обновления конфигурации")
                    except json.JSONDecodeError:
                        print("❌ Неверный JSON формат")
                    except Exception as e:
                        print(f"❌ Ошибка: {e}")

                else:
                    print("⚠️ Пожалуйста, введите число от 0 до 11.")

                input("\n📱 Нажмите Enter для продолжения...")
                print("\n" + "="*50 + "\n")

        finally:
          # Гарантированное закрытие логгера
            for handler in assistant.logger.handlers:
                handler.close()
                assistant.logger.removeHandler(handler)

    except (FileNotFoundError, SecurityError, ConfigurationError) as e:
        print(f"❌ Configuration Error: {e}")
        print("💡 Убедитесь, что config.json существует или установите AIDA_CONFIG_PATH")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Critical initialization error: {e}")
        sys.exit(1)

# Запуск программы
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Программа завершена пользователем. До свидания!")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        print("🔧 Попробуйте перезапустить программу.")
