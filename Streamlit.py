import streamlit as st
import sys
import os
import time
import json
import psutil
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
from pathlib import Path
import threading
import queue
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from contextlib import contextmanager
import traceback

# ===== КОНФИГУРАЦИЯ ЛОГИРОВАНИЯ =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== ТИПИЗАЦИЯ И МОДЕЛИ ДАННЫХ =====
class ModelStatus(Enum):
    IDLE = "idle"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"

class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    CODE_FIXING = "code_fixing"
    CODE_EXPLANATION = "code_explanation"
    FREE_MODE = "free_mode"

@dataclass
class SystemMetrics:
    """Системные метрики для мониторинга ресурсов"""
    ram_percent: float
    cpu_percent: float
    gpu_percent: Optional[float]
    disk_percent: float
    timestamp: datetime
    
@dataclass
class ModelConfiguration:
    """Конфигурация модели с валидацией"""
    dtype: Optional[str] = None
    quantization: Optional[Union[str, bool]] = None
    max_new_tokens: Optional[int] = None
    model_name: Optional[str] = None
    device: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Валидация конфигурации"""
        logger.info(f"Проверка конфигурации модели: dtype={self.dtype}, quantization={self.quantization}, max_new_tokens={self.max_new_tokens}")
        return any([
            self.dtype is not None,
            self.quantization is not None,
            self.max_new_tokens is not None
        ])

@dataclass
class InteractionRecord:
    """Запись взаимодействия с системой"""
    timestamp: str
    task: str
    result: str
    task_type: str
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

# ===== ПРОВЕРКА ЗАВИСИМОСТЕЙ =====
def check_gpu_availability() -> bool:
    """Проверка доступности GPU с обработкой ошибок"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        logger.warning("PyTorch не установлен, GPU недоступен")
        return False

gpu_available = check_gpu_availability()

def import_assistant_modules():
    """Безопасный импорт модулей ассистента"""
    try:
        from Aida import EnhancedCodeAssistant, GenerationMetrics
        return EnhancedCodeAssistant, GenerationMetrics
    except ImportError as e:
        logger.error(f"Ошибка импорта модулей AIDA: {e}")
        st.error("Не удалось импортировать модули AIDA. Убедитесь, что файл fixed_aida_code3.py находится в той же директории.")
        st.stop()

EnhancedCodeAssistant, GenerationMetrics = import_assistant_modules()

# ===== КОНФИГУРАЦИЯ STREAMLIT =====
st.set_page_config(
    page_title="🤖 Аида - AI Code Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== СТИЛИ И CSS =====
CUSTOM_CSS = """
<style>
   body {
       word-wrap: break-word;
       overflow-wrap: break-word;
    }

    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
        color: #333 !important;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-ready { background-color: #28a745; }
    .status-loading { background-color: #ffc107; animation: pulse 1s infinite; }
    .status-error { background-color: #dc3545; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .progress-bar {
        background-color: #e9ecef;
        border-radius: 4px;
        overflow: hidden;
        height: 8px;
        margin: 0.5rem 0;
    }
    
    .progress-fill {
        background-color: #007bff;
        height: 100%;
        transition: width 0.3s ease;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 4px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        color: #856404;
    }
    
    .task-section {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    
    .config-section {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-section {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #721c24;
    }

    /* НОВЫЕ СТИЛИ ДЛЯ КОРРЕКТНОГО ОТОБРАЖЕНИЯ КОДА */

    /* Основные стили для блоков кода */
    pre {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        max-width: 100% !important;
        overflow-x: auto !important;
    }

    /* Стили для inline кода */
    code {
         white-space: pre-wrap !important;
         word-wrap: break-word !important;
         overflow-wrap: break-word !important;
         max-width: 100% !important;
    }

    /* Специфичные стили для Streamlit code блоков */
    .stCodeBlock {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        max-width: 100% !important;
    }

    .stCodeBlock pre {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        max-width: 100% !important;
        overflow-x: auto !important;
    }

    .stCodeBlock code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        max-width: 100% !important;
    }

    /* Стили для элементов с классом element-container */
    .element-container pre {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        max-width: 100% !important;
        overflow-x: auto !important;
    }

    .element-container code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        max-width: 100% !important;
    }

    /* Стили для markdown кода в результатах */
    .markdown-text-container pre {
        white-space: pre-wrap !important;
         word-wrap: break-word !important;
         overflow-wrap: break-word !important;
         max-width: 100% !important;
         overflow-x: auto !important;
         font-family: 'Courier New', Courier, monospace !important;
         background-color: #f8f9fa !important;
         border: 1px solid #e9ecef !important;
         border-radius: 4px !important;
         padding: 1rem !important;
         margin: 0.5rem 0 !important;
    }

    .markdown-text-container code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        max-width: 100% !important;
        font-family: 'Courier New', Courier, monospace !important;
    }

    /* Универсальные стили для всех возможных контейнеров кода */
    [data-testid="stMarkdown"] pre,
    [data-testid="stCode"] pre,
    div[class*="code"] pre {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        max-width: 100% !important;
        overflow-x: auto !important;
    }

    [data-testid="stMarkdown"] code,
    [data-testid="stCode"] code,
    div[class*="code"] code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        max-width: 100% !important;
    }

    /* Дополнительные стили для улучшения читаемости */
    .stCodeBlock {
        font-size: 14px !important;
        line-height: 1.5 !important;
    }

    /* Стили для горизонтальной прокрутки при необходимости */
    .code-container {
        max-width: 100%;
        overflow-x: auto;
    }

    /* Responsive дизайн для мобильных устройств */
    @media (max-width: 768px) {
        pre, code {
            font-size: 12px !important;
            line-height: 1.4 !important;
        }

        .stCodeBlock {
            font-size: 12px !important;
        }
    }
</style>
"""

# ===== ОСНОВНОЙ КЛАСС ПРИЛОЖЕНИЯ =====
class EnhancedAidaWebInterface:
    """
    Production-ready веб-интерфейс для AIDA Code Assistant
    
    Архитектурные принципы:
    - Разделение ответственности через композицию
    - Comprehensive error handling
    - Type safety с использованием dataclasses
    - Кэширование для производительности
    - Graceful degradation
    """
    
    def __init__(self):
        """Инициализация с проверкой зависимостей"""
        self._initialize_session_state()
        self._setup_logging()
        self._ensure_assistant_initialized()

    def _initialize_session_state(self) -> None:
        """Инициализация состояния сессии с типизированными значениями"""
        default_states: Dict[str, Any] = {
            'assistant': None,
            'conversation_history': [],
            'current_task': None,
            'model_loaded': False,
            'user_name': None,
            'session_metrics': [],
            'last_response': None,
            'config_path': 'config.json',
            'loading_progress': 0,
            'loading_status': ModelStatus.IDLE.value,
            'loading_error': None,
            'last_resource_check': time.time(),
            'resource_warning': None,
            'loading_queue': None,
            'last_rerun_time': 0,
            'session_start': datetime.now(),
            'chat_messages': [],
            'cached_config': None,
            'config_cache_time': 0,
            'model_config_loaded': False  # Критический флаг для отслеживания состояния конфигурации
        }

        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value
   
    def load_model(self):
        """Метод для загрузки модели и обновления состояния."""
        try:
            logger.info("Начинаем загрузку модели...")
            start_time = time.time()

            success = st.session_state.assistant.load_model()
            load_time = time.time() - start_time

            # ✅ Единообразная установка состояния
            st.session_state.model_loaded = success
            st.session_state.loading_status = (
                ModelStatus.LOADED.value if success else ModelStatus.ERROR.value
            )

            if success:
                logger.info(f"Модель успешно загружена за {load_time:.2f}с")
                # Принудительное обновление кэша конфигурации
                self._get_cached_model_configuration.clear()
                st.session_state.model_config_loaded = True
            else:
                st.session_state.loading_error = "Не удалось загрузить модель"
                logger.error("Ошибка загрузки модели")

        except Exception as e:
            st.session_state.loading_status = ModelStatus.ERROR.value
            st.session_state.loading_error = str(e)
            st.session_state.model_config_loaded = False
            logger.error(f"Исключение при загрузке модели: {e}")


    def _setup_logging(self) -> None:
        """Настройка логирования для production environment"""
        if 'logger_initialized' not in st.session_state:
            logger.info("Инициализация AIDA Web Interface")
            st.session_state.logger_initialized = True

    def _ensure_assistant_initialized(self) -> None:
        """Thread-safe инициализация ассистента"""
        if st.session_state.assistant is None:
            try:
                st.session_state.assistant = EnhancedCodeAssistant(
                    st.session_state.config_path
                )
                logger.info("Ассистент успешно инициализирован")
            except Exception as e:
                logger.error(f"Ошибка инициализации ассистента: {e}")
                st.error(f"Критическая ошибка инициализации: {e}")
                st.stop()

    @contextmanager
    def _error_boundary(self, operation_name: str):
        """Context manager для безопасной обработки ошибок"""
        try:
            yield
        except Exception as e:
            logger.error(f"Ошибка в операции {operation_name}: {e}")
            st.error(f"Ошибка {operation_name}: {str(e)}")
            
            # Debug информация в development режиме
            if st.session_state.get('debug_mode', False):
                st.text_area("Stack trace:", traceback.format_exc(), height=200)

    def _validate_model_state(self) -> bool:
        """Comprehensive валидация состояния модели"""
        return (
            st.session_state.model_loaded and
            st.session_state.loading_status == ModelStatus.LOADED.value and
            st.session_state.assistant is not None and
            hasattr(st.session_state.assistant, 'load_config')
        )

    @st.cache_data(ttl=300, show_spinner=False)
    def _get_cached_model_configuration(_self) -> Optional[Dict[str, Any]]:
        """Кэшированное получение конфигурации модели"""
        try:
            if not st.session_state.assistant:
                return None
                
            # Позволяем получать конфигурацию после инициализации ассистента
            config = st.session_state.assistant.get_display_configuration()

            if isinstance(config, dict) and config:
                logger.info("Конфигурация модели успешно загружена")
                st.session_state.model_config_loaded = True
                return config
            else:
                logger.warning("Получена некорректная или пустая конфигурация модели")
                return None

        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            return None


    def _get_system_metrics(self) -> SystemMetrics:
        """Получение системных метрик с обработкой ошибок"""
        try:
            ram = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            disk = psutil.disk_usage('/')
            
            gpu_percent = None
            if gpu_available:
                try:
                    import torch
                    torch.cuda.empty_cache()
                    total = torch.cuda.get_device_properties(0).total_memory
                    used = torch.cuda.memory_allocated(0)
                    gpu_percent = (used / total) * 100
                except Exception as e:
                    logger.warning(f"Ошибка получения GPU метрик: {e}")

            return SystemMetrics(
                ram_percent=ram.percent,
                cpu_percent=cpu_percent,
                gpu_percent=gpu_percent,
                disk_percent=disk.percent,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Ошибка получения системных метрик: {e}")
            # Возвращаем dummy метрики в случае ошибки
            return SystemMetrics(0, 0, None, 0, datetime.now())

    def _check_system_resources(self) -> List[str]:
        """Проверка системных ресурсов и генерация предупреждений"""
        warnings = []
        
        try:
            metrics = self._get_system_metrics()
            
            # Проверка критических пороговых значений
            if metrics.ram_percent > 90:
                warnings.append(f"⚠️ Критическое использование RAM: {metrics.ram_percent:.1f}%")
            elif metrics.ram_percent > 80:
                warnings.append(f"⚠️ Высокое использование RAM: {metrics.ram_percent:.1f}%")

            if metrics.cpu_percent > 90:
                warnings.append(f"⚠️ Критическая нагрузка CPU: {metrics.cpu_percent:.1f}%")

            if metrics.gpu_percent and metrics.gpu_percent > 90:
                warnings.append(f"⚠️ Критическая нагрузка GPU: {metrics.gpu_percent:.1f}%")

            if metrics.disk_percent > 90:
                warnings.append(f"⚠️ Критически мало места на диске: {metrics.disk_percent:.1f}%")

        except Exception as e:
            warnings.append(f"⚠️ Ошибка мониторинга системы: {str(e)}")

        return warnings

    def _start_model_loading(self) -> None:
        """Безопасная загрузка модели с мониторингом"""
        if (st.session_state.assistant is not None
            and not st.session_state.model_loaded
            and st.session_state.loading_status != ModelStatus.LOADING.value):

            # Сброс состояний
            st.session_state.loading_progress = 0
            st.session_state.loading_error = None
            st.session_state.resource_warning = None
            st.session_state.loading_status = ModelStatus.LOADING.value
            st.session_state.model_config_loaded = False  # Сброс флага конфигурации

            # Предварительная проверка ресурсов
            pre_check_warnings = self._check_system_resources()
            if pre_check_warnings:
                st.session_state.resource_warning = "\n".join(pre_check_warnings)

            with st.spinner("🚀 Загрузка модели..."):
                self.load_model()
                try:
                    start_time = time.time()
                    logger.info("Начинаем загрузку модели...")
                    logger.info(f"Текущий статус загрузки: {st.session_state.loading_status}, Модель загружена: {st.session_state.model_loaded}")
                    success = st.session_state.assistant.load_model()
                    load_time = time.time() - start_time
                    
                    st.session_state.model_loaded = success
                    st.session_state.loading_status = (
                        ModelStatus.LOADED.value if success else ModelStatus.ERROR.value
                    )
                    
                    if success:
                        logger.info(f"Модель успешно загружена за {load_time:.2f}с")
                        # Очистка кэша конфигурации для принудительного перезапроса
                        self._get_cached_model_configuration.clear()
                        # Устанавливаем флаг готовности конфигурации
                        st.session_state.model_config_loaded = True
                    else:
                        st.session_state.loading_error = "Не удалось загрузить модель"
                        logger.error("Ошибка загрузки модели")
                        
                except Exception as e:
                    st.session_state.loading_status = ModelStatus.ERROR.value
                    st.session_state.loading_error = str(e)
                    st.session_state.model_config_loaded = False
                    logger.error(f"Исключение при загрузке модели: {e}")

            st.rerun()

    def _render_model_configuration(self) -> None:
        """Production-ready рендеринг конфигурации модели с правильной валидацией"""
        
        # КРИТИЧЕСКАЯ ВАЛИДАЦИЯ: проверяем состояние модели
        if not self._validate_model_state():
            if st.session_state.loading_status == ModelStatus.LOADING.value:
                st.info("🔄 Модель загружается, конфигурация будет доступна после завершения")
            elif st.session_state.loading_status == ModelStatus.ERROR.value:
                st.error("❌ Ошибка загрузки модели, конфигурация недоступна")
            else:
                st.info("ℹ️ Параметры модели будут отображены после загрузки")
            return

        with self._error_boundary("рендеринг конфигурации модели"):
            config_dict = self._get_cached_model_configuration()
            
            if not config_dict:
                st.warning("⚠️ Конфигурация модели недоступна")
                return

            config = ModelConfiguration(**{
                k: v for k, v in config_dict.items() 
                if k in ModelConfiguration.__annotations__
            })

            if not config.is_valid():
                st.warning("⚠️ Конфигурация модели неполная")
                return

            st.markdown('<div class="config-section">', unsafe_allow_html=True)
            st.subheader("⚙️ Параметры модели")
            
            # Основные параметры с валидацией
            if config.dtype:
                st.write(f"**Тип данных:** `{config.dtype}`")
            
            if config.quantization is not None:
                quant_status = "✅ " + str(config.quantization) if config.quantization else "❌ Отключена"
                st.write(f"**Квантизация:** {quant_status}")
                
            if config.max_new_tokens:
                st.write(f"**Макс. токены:** {config.max_new_tokens:,}")
                
            if config.model_name:
                st.write(f"**Модель:** {config.model_name}")
                
            if config.device:
                st.write(f"**Устройство:** {config.device}")

            # Индикатор успешности загрузки конфигурации
            st.success("✅ Конфигурация модели загружена")

            # Debug секция для development
            if st.session_state.get('debug_mode', False):
                with st.expander("🔧 Debug конфигурации"):
                    st.json(config_dict)
                    
            st.markdown('</div>', unsafe_allow_html=True)

    def _render_system_monitoring(self) -> None:
        """Enhanced система мониторинга ресурсов"""
        with self._error_boundary("мониторинг системы"):
            metrics = self._get_system_metrics()
            
            # Основные метрики
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "RAM", 
                    f"{metrics.ram_percent:.1f}%",
                    help=f"Обновлено: {metrics.timestamp.strftime('%H:%M:%S')}"
                )
                
            with col2:
                st.metric("CPU", f"{metrics.cpu_percent:.1f}%")
                
            with col3:
                if metrics.gpu_percent is not None:
                    st.metric("GPU", f"{metrics.gpu_percent:.1f}%")
                else:
                    st.metric("GPU", "N/A")
                    
            with col4:
                st.metric("Диск", f"{metrics.disk_percent:.1f}%")

            # Предупреждения
            warnings = self._check_system_resources()
            if warnings:
                st.markdown(f"""
                <div class="warning-box">
                    {"<br>".join(warnings)}
                </div>
                """, unsafe_allow_html=True)

    def render_header(self) -> None:
        """Рендеринг заголовка приложения"""
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        st.markdown("""
        <div class="main-header">
            <h1>🤖 Аида - AI Code Assistant</h1>
            <p>Production-Ready</p>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self) -> None:
        """Enhanced боковая панель с comprehensive monitoring"""
        with st.sidebar:
            st.header("🎛️ Панель управления")

            # Статус модели с типизированными состояниями
            status_config = {
                ModelStatus.IDLE.value: ("Ожидание загрузки", "status-ready"),
                ModelStatus.LOADING.value: ("Загрузка...", "status-loading"),
                ModelStatus.LOADED.value: ("Модель загружена", "status-ready"),
                ModelStatus.ERROR.value: (f"Ошибка: {st.session_state.loading_error or 'Неизвестная ошибка'}", "status-error")
            }

            status_text, status_class = status_config[st.session_state.loading_status]

            st.markdown(f"""
            <div class="metric-card">
                <span class="status-indicator {status_class}"></span>
                <strong>Статус:</strong> {status_text}
            </div>
            """, unsafe_allow_html=True)

            # Progress bar для загрузки
            if st.session_state.loading_status == ModelStatus.LOADING.value:
                st.markdown(f"""
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {st.session_state.loading_progress}%"></div>
                </div>
                <div style="text-align: center; font-size: 0.8rem;">
                    {st.session_state.loading_progress}%
                </div>
                """, unsafe_allow_html=True)

            # Пользовательская идентификация
            if not st.session_state.user_name:
                name = st.text_input("👤 Как вас зовут?", placeholder="Введите ваше имя")
                if name:
                    st.session_state.user_name = name
                    if st.session_state.assistant:
                        st.session_state.assistant.user_name = name
                    logger.info(f"Пользователь идентифицирован: {name}")
                    st.rerun()
            else:
                st.success(f"👋 Привет, {st.session_state.user_name}!")

            # Управление моделью
            if st.session_state.loading_status != ModelStatus.LOADING.value:
                if not st.session_state.model_loaded:
                    if st.button("🚀 Загрузить модель", type="primary"):
                        self._start_model_loading()
                else:
                    st.success("✅ Модель готова к работе!")

            # Отображение предупреждений
            if st.session_state.resource_warning:
                st.markdown(f"""
                <div class="warning-box">
                    {st.session_state.resource_warning}
                </div>
                """, unsafe_allow_html=True)

            # Мониторинг ресурсов
            st.subheader("📊 Мониторинг ресурсов")
            self._render_system_monitoring()

            # Конфигурация модели - ИСПРАВЛЕННАЯ ЛОГИКА
            self._render_model_configuration()

            # Статистика сессии
            self._render_session_statistics()

            # Управление сессией
            self._render_session_controls()

    def _render_session_statistics(self) -> None:
        """Рендеринг статистики сессии"""
        if st.session_state.conversation_history:
            st.subheader("📊 Статистика сессии")
            
            total_tasks = len(st.session_state.conversation_history)
            st.metric("Всего задач", total_tasks)

            # Анализ типов задач
            task_types = {}
            for interaction in st.session_state.conversation_history:
                task_type = interaction.get('task_type', 'unknown')
                task_types[task_type] = task_types.get(task_type, 0) + 1

            for task_type, count in task_types.items():
                display_name = task_type.replace('_', ' ').title()
                st.metric(display_name, count)

            # Время сессии
            session_duration = datetime.now() - st.session_state.session_start
            st.metric("Время сессии", f"{int(session_duration.total_seconds() / 60)} мин")

    def _render_session_controls(self) -> None:
        """Элементы управления сессией"""
        if st.session_state.conversation_history:
            if st.button("🧹 Очистить историю", type="secondary"):
                st.session_state.conversation_history = []
                st.session_state.chat_messages = []
                if st.session_state.assistant:
                    st.session_state.assistant.conversation_history = []
                logger.info("История сессии очищена")
                st.rerun()

        # Debug режим toggle
        st.session_state.debug_mode = st.checkbox("🔧 Debug режим", value=st.session_state.get('debug_mode', False))

    def render_main_interface(self) -> None:
        """Главный интерфейс с проверкой готовности модели"""
        if not st.session_state.model_loaded:
            st.markdown("""
            <div class="warning-box">
                <h3>⚠️ Модель не загружена</h3>
                <p>Для начала работы загрузите модель в боковой панели.</p>
                <p><strong>Совет:</strong> Убедитесь, что у вас достаточно системных ресурсов для загрузки модели.</p>
            </div>
            """, unsafe_allow_html=True)
            return

        # Интерфейс с вкладками
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "✨ Генерация кода",
            "🔍 Анализ кода", 
            "🛠️ Исправление кода",
            "📚 Объяснение кода",
            "🧠 Свободный режим",
            "📊 Мониторинг",
            "⚙️ Настройки"
        ])

        with tab1:
            self._render_code_generation()
        with tab2:
            self._render_code_analysis()
        with tab3:
            self._render_code_fixing()
        with tab4:
            self._render_code_explanation()
        with tab5:
            self._render_free_mode()
        with tab6:
            self._render_monitoring()
        with tab7:
            self._render_runtime_settings()

    def _render_code_generation(self) -> None:
        """Enhanced интерфейс генерации кода"""
        st.markdown('<div class="task-section">', unsafe_allow_html=True)
        st.subheader("✨ Генерация кода")
        st.write("Опишите задачу, и Аида создаст качественный, production-ready код")

        col1, col2 = st.columns([3, 1])

        with col1:
            task = st.text_area(
                "📝 Описание задачи:",
                placeholder="Например: создать асинхронную функцию для обработки HTTP запросов с retry логикой и comprehensive error handling",
                height=120,
                help="Чем детальнее описание, тем качественнее результат"
            )

        with col2:
            languages = [
                "python", "javascript", "typescript", "java", "c++", "c", "c#",
                "go", "rust", "php", "ruby", "swift", "kotlin", "scala",
                "html", "css", "sql", "bash", "r", "matlab", "lua"
            ]
            language = st.selectbox("🔧 Язык программирования:", languages)
            
            # Дополнительные опции
            include_tests = st.checkbox("🧪 Включить тесты", help="Генерировать unit тесты для кода")
            include_docs = st.checkbox("📚 Включить документацию", help="Добавить docstrings и комментарии")

        if st.button("🚀 Генерировать код", type="primary", key="gen_code"):
            if task.strip():
                with self._error_boundary("генерация кода"):
                    with st.spinner("🤖 Аида генерирует код..."):
                        start_time = time.time()
                        
                        # Расширенный промпт с опциями
                        enhanced_task = task
                        if include_tests:
                            enhanced_task += "\n\n🧪 ВАЖНО: Также включи comprehensive unit тесты с использованием подходящего фреймворка для тестирования."
                        if include_docs:
                            enhanced_task += "\n\n📚 ВАЖНО: Добавь детальную документацию, docstrings и комментарии, объясняющие логику работы."
                        
                        result = st.session_state.assistant.generate_code(enhanced_task, language)
                        duration = (time.time() - start_time) * 1000
                        
                        self._save_interaction(
                            task, result, TaskType.CODE_GENERATION.value, duration
                        )

                        st.markdown("### 💻 Результат:")
                        st.code(result, language=language.lower())

                        # Дополнительная информация
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.info(f"⏱️ Время генерации: {duration:.0f}мс")
                        with col_info2:
                            lines_count = len(result.split('\n'))
                            st.info(f"📏 Строк кода: {lines_count}")

                        # Показываем примененные опции
                        if include_tests or include_docs:
                            options_applied = []
                            if include_tests:
                                options_applied.append("🧪 Тесты")
                            if include_docs:
                                options_applied.append("📚 Документация")
                            st.success(f"✅ Применены опции: {', '.join(options_applied)}")

                        # Кнопка скачивания
                        file_extension = self._get_file_extension(language)
                        st.download_button(
                            "📥 Скачать код",
                            result,
                            file_name=f"generated_code.{file_extension}",
                            mime="text/plain"
                        )
            else:
                st.warning("⚠️ Пожалуйста, опишите задачу")

        st.markdown('</div>', unsafe_allow_html=True)

    def _render_code_analysis(self) -> None:
        """Enhanced интерфейс анализа кода"""
        st.markdown('<div class="task-section">', unsafe_allow_html=True)
        st.subheader("🔍 Анализ кода")
        st.write("Загрузите код для профессионального анализа качества, безопасности и производительности")

        col1, col2 = st.columns([3, 1])

        with col1:
            code_input_method = st.radio(
                "Способ ввода кода:",
                ["📝 Текстовое поле", "📁 Загрузка файла"],
                horizontal=True
            )

            if code_input_method == "📝 Текстовое поле":
                code = st.text_area(
                    "Код для анализа:", 
                    height=250, 
                    placeholder="Вставьте ваш код здесь...",
                    help="Поддерживаются все основные языки программирования"
                )
            else:
                uploaded_file = st.file_uploader(
                    "Выберите файл с кодом:", 
                    type=['py', 'js', 'ts', 'java', 'cpp', 'c', 'cs', 'go', 'rs', 'php', 'rb', 'html', 'css'],
                    help="Максимальный размер файла: 200MB"
                )
                code = ""
                if uploaded_file:
                    try:
                        # Проверка размера файла
                        if uploaded_file.size > 200 * 1024 * 1024:  # 200MB
                            st.error("❌ Файл слишком большой. Максимальный размер: 200MB")
                        else:
                            code = uploaded_file.read().decode('utf-8')
                            st.success(f"✅ Файл загружен: {uploaded_file.name} ({len(code)} символов)")
                            
                            # Предварительный просмотр
                            with st.expander("👁️ Предварительный просмотр"):
                                preview_lines = code.split('\n')[:20]
                                st.code('\n'.join(preview_lines), language='python')
                                if len(code.split('\n')) > 20:
                                    st.info(f"... и еще {len(code.splitlines()) - 20} строк")
                                    
                    except UnicodeDecodeError:
                        st.error("❌ Ошибка кодировки файла. Убедитесь, что файл в UTF-8")
                    except Exception as e:
                        st.error(f"❌ Ошибка чтения файла: {str(e)}")

        with col2:
            languages = ["python", "javascript", "typescript", "java", "c++", "c#", "go", "rust", "php", "ruby"]
            language = st.selectbox("🔧 Язык:", languages, key="analyze_lang")
            
            # Опции анализа
            st.write("**Опции анализа:**")
            check_security = st.checkbox("🔒 Анализ безопасности", value=True)
            check_performance = st.checkbox("⚡ Анализ производительности", value=True)
            check_style = st.checkbox("🎨 Анализ стиля кода", value=True)
            check_complexity = st.checkbox("🧮 Анализ сложности", value=True)

        if st.button("🔬 Анализировать", type="primary", key="analyze_code"):
            if code.strip():
                with self._error_boundary("анализ кода"):
                    with st.spinner("🤖 Аида анализирует код..."):
                        start_time = time.time()
                        
                        # Формирование расширенного запроса с учётом опций
                        analysis_options = []
                        if check_security:
                            analysis_options.append("🔒 безопасность и поиск уязвимостей")
                        if check_performance:
                            analysis_options.append("⚡ производительность и возможности оптимизации")
                        if check_style:
                            analysis_options.append("🎨 стиль кода и соответствие стандартам")
                        if check_complexity:
                            analysis_options.append("🧮 сложность кода и архитектура")
                        
                        # Создаем полный запрос на анализ
                        if analysis_options:
                            analysis_prompt = f"Выполни детальный анализ следующего кода на языке {language} с особым фокусом на:\n\n{chr(10).join(analysis_options)}\n\nКод для анализа:\n\n{code}"
                        else:
                            analysis_prompt = f"Выполни общий анализ следующего кода на языке {language}:\n\n{code}"
                        # Используем расширенный промпт
                        result = st.session_state.assistant.analyze_code(analysis_prompt, language)
                        duration = (time.time() - start_time) * 1000

                        self._save_interaction(
                            f"Анализ кода ({language})", result, TaskType.CODE_ANALYSIS.value, duration
                        )

                        st.markdown("### 📋 Результат анализа:")
                        st.markdown(result)
                        
                        # Метрики анализа
                        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                        with col_metrics1:
                            st.metric("⏱️ Время анализа", f"{duration:.0f}мс")
                        with col_metrics2:
                            lines_analyzed = len(code.split('\n'))
                            st.metric("📏 Строк проанализировано", lines_analyzed)
                        with col_metrics3:
                            chars_analyzed = len(code)
                            st.metric("📝 Символов", f"{chars_analyzed:,}")

                        # Показываем примененные опции анализа
                        if analysis_options:
                            st.success(f"✅ Применены опции анализа: {', '.join(analysis_options)}")
            else:
                st.warning("⚠️ Пожалуйста, введите код для анализа")

        st.markdown('</div>', unsafe_allow_html=True)

    def _render_code_fixing(self) -> None:
        """Enhanced интерфейс исправления кода"""
        st.markdown('<div class="task-section">', unsafe_allow_html=True)
        st.subheader("🛠️ Исправление и улучшение кода")
        st.write("Аида найдет и исправит ошибки, оптимизирует производительность и улучшит качество кода")

        col1, col2 = st.columns([3, 1])

        with col1:
            code = st.text_area(
                "Код для исправления:", 
                height=250, 
                placeholder="Вставьте код с ошибками или требующий улучшения...",
                help="Опишите проблему в комментариях, если она не очевидна"
            )
            
            # Дополнительное описание проблемы
            problem_description = st.text_area(
                "📝 Описание проблемы (опционально):",
                height=80,
                placeholder="Опишите, какие именно проблемы вы наблюдаете или что нужно улучшить..."
            )

        with col2:
            languages = ["python", "javascript", "typescript", "java", "c++", "c#", "go", "rust", "php", "ruby"]
            language = st.selectbox("🔧 Язык:", languages, key="fix_lang")
            
            # Опции исправления
            st.write("**Типы исправлений:**")
            fix_bugs = st.checkbox("🐛 Исправить баги", value=True)
            optimize_performance = st.checkbox("⚡ Оптимизация производительности", value=True)
            improve_readability = st.checkbox("📖 Улучшить читаемость", value=True)
            add_error_handling = st.checkbox("🛡️ Добавить обработку ошибок", value=True)
            modernize_code = st.checkbox("🔄 Модернизировать код", value=False)

        if st.button("🔧 Исправить код", type="primary", key="fix_code"):
            if code.strip():
                with self._error_boundary("исправление кода"):
                    with st.spinner("🤖 Аида исправляет код..."):
                        start_time = time.time()
                        
                        # Формирование детального запроса с учётом опций
                        fix_options = []
                        if fix_bugs:
                            fix_options.append("🐛 исправление багов и логических ошибок")
                        if optimize_performance:
                            fix_options.append("⚡ оптимизация производительности и алгоритмов")
                        if improve_readability:
                            fix_options.append("📖 улучшение читаемост, структуры и именования")
                        if add_error_handling:
                            fix_options.append("🛡️ добавление comprehensive error handling и валидации")
                        if modernize_code:
                            fix_options.append("🔄 модернизация с использованием современных практик и паттернов")
                        
                        # Создаем полный запрос на исправление
                        fix_prompt = f"Исправь и улучши следующий код на языке {language}"
                        if fix_options:
                            fix_prompt += f" с фокусом на:\n\n{chr(10).join(fix_options)}"

                        if problem_description.strip():
                            fix_prompt += f"\n\nОписание проблемы от пользователя:\n{problem_description}"

                        fix_prompt += f"\n\nКод для исправления:\n\n{code}"
                        
                        # Используем расширенный промпт
                        result = st.session_state.assistant.fix_code(fix_prompt, language)
                        duration = (time.time() - start_time) * 1000
                        
                        self._save_interaction(
                            f"Исправление кода ({language})", result, TaskType.CODE_FIXING.value, duration
                        )

                        st.markdown("### ✅ Исправленный код:")
                        st.markdown(result)

                        # Статистика исправлений
                        col_stats1, col_stats2 = st.columns(2)
                        with col_stats1:
                            st.info(f"⏱️ Время исправления: {duration:.0f}мс")
                        with col_stats2:
                            original_lines = len(code.split('\n'))
                            st.info(f"📏 Исходных строк: {original_lines}")

                        # Показываем примененные опции исправления
                        if fix_options:
                            st.success(f"✅ Применены опции исправления: {', '.join(fix_options)}")

                        # Извлечение блоков кода для скачивания
                        import re
                        code_blocks = re.findall(r'```[\w]*\n(.*?)\n```', result, re.DOTALL)
                        if code_blocks:
                            file_extension = self._get_file_extension(language)
                            st.download_button(
                                "📥 Скачать исправленный код",
                                code_blocks[0],
                                file_name=f"fixed_code.{file_extension}",
                                mime="text/plain"
                            )
            else:
                st.warning("⚠️ Пожалуйста, введите код для исправления")

        st.markdown('</div>', unsafe_allow_html=True)

    def _render_code_explanation(self) -> None:
        """Enhanced интерфейс объяснения кода"""
        st.markdown('<div class="task-section">', unsafe_allow_html=True)
        st.subheader("📚 Объяснение кода")
        st.write("Получите детальное, структурированное объяснение работы кода с примерами")

        col1, col2 = st.columns([3, 1])

        with col1:
            code = st.text_area(
                "Код для объяснения:", 
                height=250, 
                placeholder="Вставьте код, который нужно объяснить...",
                help="Чем сложнее код, тем детальнее будет объяснение"
            )

        with col2:
            detail_levels = {
                "Кратко": "краткое объяснение основной логики",
                "Подробно": "подробное объяснение с примерами",
                "Для новичков": "объяснение для начинающих разработчиков",
                "Экспертно": "техническое объяснение с архитектурными деталями",
                "С примерами": "объяснение с примерами использования"
            }
            detail_level = st.selectbox("📖 Уровень детализации:", list(detail_levels.keys()))
            
            # Дополнительные опции
            include_diagrams = st.checkbox("📊 Включить диаграммы", help="Добавить ASCII диаграммы для визуализации")
            explain_complexity = st.checkbox("🧮 Объяснить сложность", help="Анализ временной и пространственной сложности")

        if st.button("📖 Объяснить код", type="primary", key="explain_code"):
            if code.strip():
                with self._error_boundary("объяснение кода"):
                    with st.spinner("🤖 Аида объясняет код..."):
                        start_time = time.time()
                        
                        # == Расширенный запрос с учетом всех опций ==
                        enhanced_request = f"Дай {detail_levels[detail_level]} следующего кода"

                        additional_options = []
                        if include_diagrams:
                            additional_options.append("📊 включи ASCII диаграммы и схемы для визуализации логики")
                        if explain_complexity:
                            additional_options.append("🧮 проанализируй временную и пространственную сложность алгоритмов")

                        if additional_options:
                            enhanced_request += f", при этом обязательно {', а также '.join(additional_options)}"

                        enhanced_request += f":\n\n{code}"
                        
                        result = st.session_state.assistant.explain_code(code, enhanced_request)
                        duration = (time.time() - start_time) * 1000
                        
                        self._save_interaction(
                            f"Объяснение кода ({detail_level})", result, TaskType.CODE_EXPLANATION.value, duration
                        )

                        st.markdown("### 💡 Объяснение:")
                        st.markdown(result)
                        
                        # Метрики объяснения
                        col_exp1, col_exp2, col_exp3 = st.columns(3)
                        with col_exp1:
                            st.metric("⏱️ Время объяснения", f"{duration:.0f}мс")
                        with col_exp2:
                            explanation_words = len(result.split())
                            st.metric("📝 Слов в объяснении", explanation_words)
                        with col_exp3:
                            code_lines = len(code.split('\n'))
                            st.metric("📏 Строк кода", code_lines)

                        # Показываем примененные опции объяснения
                        applied_options = [f"📖 {detail_level}"]
                        if include_diagrams:
                            applied_options.append("📊 Диаграммы")
                        if explain_complexity:
                            applied_options.append("🧮 Анализ сложности")

                        st.success(f"✅ Применены опции: {', '.join(applied_options)}")
            else:
                st.warning("⚠️ Пожалуйста, введите код для объяснения")

        st.markdown('</div>', unsafe_allow_html=True)

    def _render_free_mode(self) -> None:
        """Enhanced свободный режим общения"""
        st.markdown('<div class="task-section">', unsafe_allow_html=True)
        st.subheader("🧠 Свободный режим")
        st.write("Общайтесь с Аидой на естественном языке по любым вопросам программирования")

        # Предустановленные промпты для быстрого доступа
        with st.expander("💡 Быстрые вопросы"):
            quick_prompts = [
                "Объясни разницу между синхронным и асинхронным программированием",
                "Какие паттерны проектирования лучше использовать для веб-API?",
                "Как оптимизировать производительность базы данных?",
                "Расскажи о best practices для code review",
                "Какие инструменты лучше использовать для CI/CD?"
            ]
            
            for i, prompt in enumerate(quick_prompts):
                if st.button(f"💭 {prompt}", key=f"quick_{i}"):
                    st.session_state.chat_messages.append({"role": "user", "content": prompt})
                    st.rerun()

        # История чата с улучшенным дизайном
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Дополнительные действия для сообщений ассистента
                    if message["role"] == "assistant" and len(message["content"]) > 100:
                        col1, col2, col3 = st.columns([1, 1, 8])
                        with col1:
                            if st.button("📋", key=f"copy_{i}", help="Копировать ответ"):
                                st.text_area("Скопируйте текст:", message["content"], height=100, key=f"copy_area_{i}")
                        with col2:
                            if st.button("💾", key=f"save_{i}", help="Сохранить как файл"):
                                st.download_button(
                                    "Скачать",
                                    message["content"],
                                    file_name=f"aida_response_{i}.md",
                                    mime="text/markdown",
                                    key=f"download_{i}"
                                )

        # Ввод чата с улучшенным интерфейсом
        if prompt := st.chat_input("Напишите ваш вопрос... (поддерживаются вопросы по программированию, архитектуре, best practices)"):
            # Добавляем сообщение пользователя
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            # Генерация ответа
            with st.chat_message("assistant"):
                with st.spinner("🤖 Аида обдумывает ответ..."):
                    try:
                        start_time = time.time()
                        response = st.session_state.assistant.generate_response(prompt, "free_mode")
                        duration = (time.time() - start_time) * 1000
                        
                        st.markdown(response)
                        st.session_state.chat_messages.append({"role": "assistant", "content": response})
                        
                        self._save_interaction(prompt, response, TaskType.FREE_MODE.value, duration)
                        
                        # Показать время ответа
                        st.caption(f"⏱️ Время ответа: {duration:.0f}мс")
                        
                    except Exception as e:
                        error_msg = f"❌ Ошибка генерации ответа: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})

        # Управление чатом
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state.chat_messages and st.button("🧹 Очистить чат"):
                st.session_state.chat_messages = []
                st.rerun()
                
        with col2:
            if st.session_state.chat_messages and st.button("📄 Экспорт чата"):
                chat_export = "\n\n".join([
                    f"**{msg['role'].upper()}:** {msg['content']}" 
                    for msg in st.session_state.chat_messages
                ])
                st.download_button(
                    "💾 Скачать чат",
                    chat_export,
                    file_name=f"aida_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
                
        with col3:
            total_messages = len(st.session_state.chat_messages)
            if total_messages > 0:
                st.metric("💬 Сообщений", total_messages)
                
        with col4:
            if st.session_state.chat_messages:
                total_chars = sum(len(msg['content']) for msg in st.session_state.chat_messages)
                st.metric("📝 Символов", f"{total_chars:,}")

        st.markdown('</div>', unsafe_allow_html=True)

    def _render_monitoring(self) -> None:
        """Enhanced мониторинг и аналитика"""
        st.subheader("📊 Comprehensive мониторинг и аналитика")

        if not st.session_state.conversation_history:
            st.markdown("""
            <div class="metric-card">
                <h4>📈 Добро пожаловать в систему мониторинга!</h4>
                <p>Статистика и аналитика будут доступны после выполнения задач.</p>
                <p><strong>Что отслеживается:</strong></p>
                <ul>
                    <li>🔢 Количество и типы выполненных задач</li>
                    <li>⏱️ Время выполнения операций</li> 
                    <li>📊 Производительность системы</li>
                    <li>🎯 Паттерны использования</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            return

        try:
            # Общие метрики
            col1, col2, col3, col4 = st.columns(4)

            total_tasks = len(st.session_state.conversation_history)
            
            # Анализ типов задач
            task_types = {}
            total_duration = 0
            successful_tasks = 0
            
            for interaction in st.session_state.conversation_history:
                task_type = interaction.get('task_type', 'unknown')
                task_types[task_type] = task_types.get(task_type, 0) + 1
                
                if 'duration_ms' in interaction:
                    total_duration += interaction['duration_ms']
                if interaction.get('success', True):
                    successful_tasks += 1

            with col1:
                st.metric("📋 Всего задач", total_tasks)

            with col2:
                success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
                st.metric("✅ Успешность", f"{success_rate:.1f}%")

            with col3:
                most_common_task = max(task_types, key=task_types.get) if task_types else "N/A"
                display_task = most_common_task.replace('_', ' ').title()
                st.metric("🎯 Популярная задача", display_task)

            with col4:
                avg_duration = total_duration / total_tasks if total_tasks > 0 else 0
                st.metric("⏱️ Ср. время", f"{avg_duration:.0f}мс")

            # Дополнительные метрики
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                session_duration = datetime.now() - st.session_state.session_start
                st.metric("🕐 Время сессии", f"{int(session_duration.total_seconds() / 60)} мин")
                
            with col6:
                tasks_per_hour = total_tasks / (session_duration.total_seconds() / 3600) if session_duration.total_seconds() > 0 else 0
                st.metric("📈 Задач/час", f"{tasks_per_hour:.1f}")
                
            with col7:
                chat_messages = len(st.session_state.chat_messages)
                st.metric("💬 Сообщений в чате", chat_messages)
                
            with col8:
                if hasattr(st.session_state.assistant, 'get_generation_statistics'):
                    gen_stats = st.session_state.assistant.get_generation_statistics()
                    if gen_stats and 'total_tokens_generated' in gen_stats:
                        st.metric("🔤 Токенов", f"{gen_stats['total_tokens_generated']:,}")
                    else:
                        st.metric("🔤 Токенов", "N/A")
                else:
                    st.metric("🔤 Токенов", "N/A")

            # Визуализации
            if task_types:
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    # График распределения типов задач
                    fig_pie = px.pie(
                        values=list(task_types.values()),
                        names=[name.replace('_', ' ').title() for name in task_types.keys()],
                        title="📊 Распределение типов задач",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col_chart2:
                    # График производительности по типам задач
                    if any('duration_ms' in interaction for interaction in st.session_state.conversation_history):
                        task_performance = {}
                        for interaction in st.session_state.conversation_history:
                            if 'duration_ms' in interaction:
                                task_type = interaction.get('task_type', 'unknown')
                                if task_type not in task_performance:
                                    task_performance[task_type] = []
                                task_performance[task_type].append(interaction['duration_ms'])
                        
                        # Средняя производительность по типам
                        avg_performance = {
                            task_type.replace('_', ' ').title(): sum(durations) / len(durations)
                            for task_type, durations in task_performance.items()
                        }
                        
                        fig_bar = px.bar(
                            x=list(avg_performance.keys()),
                            y=list(avg_performance.values()),
                            title="⚡ Средняя скорость выполнения (мс)",
                            color=list(avg_performance.values()),
                            color_continuous_scale="viridis"
                        )
                        fig_bar.update_layout(showlegend=False)
                        st.plotly_chart(fig_bar, use_container_width=True)

            # Временная активность
            df_history = pd.DataFrame(st.session_state.conversation_history)
            if not df_history.empty and 'timestamp' in df_history.columns:
                df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
                df_history['hour'] = df_history['timestamp'].dt.hour
                df_history['minute_bucket'] = (df_history['timestamp'].dt.minute // 10) * 10

                col_time1, col_time2 = st.columns(2)
                
                with col_time1:
                    # Почасовая активность
                    hourly_activity = df_history.groupby('hour').size().reset_index(name='count')
                    fig_timeline = px.bar(
                        hourly_activity,
                        x='hour',
                        y='count',
                        title="⏰ Активность по часам",
                        color='count',
                        color_continuous_scale="blues"
                    )
                    fig_timeline.update_layout(showlegend=False)
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                with col_time2:
                    # Тренд производительности
                    if 'duration_ms' in df_history.columns:
                        df_history['task_index'] = range(len(df_history))
                        fig_trend = px.line(
                            df_history,
                            x='task_index',
                            y='duration_ms',
                            title="📈 Тренд производительности",
                            labels={'task_index': 'Номер задачи', 'duration_ms': 'Время выполнения (мс)'}
                        )
                        # Добавляем линию тренда
                        fig_trend.add_scatter(
                            x=df_history['task_index'],
                            y=df_history['duration_ms'].rolling(window=3, center=True).mean(),
                            mode='lines',
                            name='Скользящее среднее',
                            line=dict(color='red', dash='dash')
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)

            # Системные метрики производительности
            st.subheader("🚀 Метрики производительности модели")
            
            if hasattr(st.session_state.assistant, 'get_generation_statistics'):
                gen_stats = st.session_state.assistant.get_generation_statistics()
                if gen_stats and 'error' not in gen_stats:
                    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

                    with perf_col1:
                        latency = gen_stats.get('average_latency_ms', 0)
                        st.metric("⏱️ Ср. латентность", f"{latency:.1f} мс")

                    with perf_col2:
                        throughput = gen_stats.get('average_throughput_tokens_per_second', 0)
                        st.metric("🚀 Пропускная способность", f"{throughput:.1f} ток/с")

                    with perf_col3:
                        total_tokens = gen_stats.get('total_tokens_generated', 0)
                        st.metric("🔤 Всего токенов", f"{total_tokens:,}")

                    with perf_col4:
                        avg_tokens_per_task = total_tokens / total_tasks if total_tasks > 0 else 0
                        st.metric("📊 Токенов/задача", f"{avg_tokens_per_task:.0f}")

                    # Детальные метрики производительности
                    with st.expander("🔬 Детальная статистика производительности"):
                        perf_details_col1, perf_details_col2 = st.columns(2)

                        with perf_details_col1:
                            st.write("**Метрики латентности:**")
                            st.write(f"- Средняя: {gen_stats.get('average_latency_ms', 0):.2f}мс")
                            st.write(f"- P95: {gen_stats.get('p95_latency_ms', 0):.2f}мс")
                            st.write(f"- P99: {gen_stats.get('p99_latency_ms', 0):.2f}мс")

                        with perf_details_col2:
                            st.write("**Метрики пропускной способности:**")
                            st.write(f"- Средняя: {gen_stats.get('average_throughput_tokens_per_second', 0):.2f} ток/с")
                            st.write(f"- Пиковая: {gen_stats.get('peak_throughput_tokens_per_second', 0):.2f} ток/с")
                            st.write(f"- Всего обработано: {gen_stats.get('total_tokens_generated', 0):,} токенов")
                else:
                    st.info("📊 Метрики производительности будут доступны после выполнения задач с загруженной моделью")

            # Детальная история задач
            st.subheader("📋 История выполненных задач")

            if st.session_state.conversation_history:
                # Фильтры для истории
                filter_col1, filter_col2, filter_col3 = st.columns(3)

                with filter_col1:
                    task_filter = st.selectbox(
                        "Фильтр по типу:",
                        ["Все"] + list(task_types.keys()),
                        format_func=lambda x: x.replace('_', ' ').title() if x != "Все" else "Все"
                    )

                with filter_col2:
                    show_errors_only = st.checkbox("Показать только ошибки")

                with filter_col3:
                    max_records = st.number_input("Максимум записей:", min_value=5, max_value=100, value=20)

                # Фильтрация данных
                filtered_history = st.session_state.conversation_history.copy()

                if task_filter != "Все":
                    filtered_history = [h for h in filtered_history if h.get('task_type') == task_filter]

                if show_errors_only:
                    filtered_history = [h for h in filtered_history if not h.get('success', True)]

                filtered_history = filtered_history[-max_records:]

                # Отображение истории в виде таблицы
                if filtered_history:
                    history_df = pd.DataFrame([
                        {
                            'Время': pd.to_datetime(h['timestamp']).strftime('%H:%M:%S'),
                            'Тип задачи': h.get('task_type', 'unknown').replace('_', ' ').title(),
                            'Задача': h.get('task', '')[:50] + '...' if len(h.get('task', '')) > 50 else h.get('task', ''),
                            'Статус': '✅' if h.get('success', True) else '❌',
                            'Время выполнения (мс)': f"{h.get('duration_ms', 0):.0f}" if 'duration_ms' in h else 'N/A'
                        }
                        for h in filtered_history
                    ])

                    st.dataframe(
                        history_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Экспорт истории
                    if st.button("📄 Экспорт истории в CSV"):
                        csv_data = pd.DataFrame(st.session_state.conversation_history).to_csv(index=False)
                        st.download_button(
                            "💾 Скачать CSV",
                            csv_data,
                            file_name=f"aida_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("Нет записей, соответствующих выбранным фильтрам")

        except Exception as e:
            logger.error(f"Ошибка отображения мониторинга: {e}")
            st.error(f"Ошибка отображения мониторинга: {str(e)}")

            # Debug информация
            if st.session_state.get('debug_mode', False):
                with st.expander("🔧 Debug информация"):
                    st.text_area("Traceback:", traceback.format_exc(), height=200)



    def _render_runtime_settings(self) -> None:
        """Интерфейс для изменения конфигурации в реальном времени."""
        st.markdown('<div class="task-section">', unsafe_allow_html=True)
        st.subheader("⚙️ Настройки генерации в реальном времени")
        st.write("Здесь вы можете изменить параметры модели без перезагрузки. Изменения будут применены к следующим запросам.")

        # Проверяем, что ассистент инициализирован
        if not st.session_state.get('assistant'):
            st.warning("Ассистент не инициализирован.")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        try:
            # Выбор секции для редактирования
            config_sections = ["generation", "performance", "monitoring"]
            section_to_edit = st.selectbox(
                "Выберите секцию конфигурации для редактирования:",
                config_sections
            )

            # Отображение текущих настроек
            if hasattr(st.session_state.assistant, section_to_edit):
                current_config = getattr(st.session_state.assistant, section_to_edit)

                with st.expander("Показать текущие настройки"):
                    st.json(current_config)

                # Поле для ввода JSON с обновлениями
                st.write("**Введите обновления в формате JSON:**")
                update_json_str = st.text_area(
                    "JSON с изменениями:",
                    height=150,
                    placeholder='{\n  "temperature": 0.8,\n  "max_new_tokens": 2000\n}',
                    help="Укажите только те параметры, которые хотите изменить."
                )

                if st.button("🚀 Применить изменения", key=f"apply_{section_to_edit}"):
                    if update_json_str:
                        try:
                            updates = json.loads(update_json_str)
                            if not isinstance(updates, dict):
                                st.error("❌ Введенные данные должны быть JSON-объектом (словарем).")
                            else:
                                success = st.session_state.assistant.update_config_runtime(section_to_edit, updates)
                                if success:
                                    st.success(f"✅ Конфигурация '{section_to_edit}' успешно обновлена!")
                                    st.info("Изменения вступят в силу только для текущей генерации.")
                                    # Можно добавить принудительный rerun для обновления UI
                                    # st.rerun()
                                else:
                                    st.error("❌ Не удалось обновить конфигурацию. Проверьте логи.")

                        except json.JSONDecodeError:
                            st.error("❌ Ошибка: введенный текст не является корректным JSON.")
                        except Exception as e:
                            st.error(f"❌ Произошла ошибка при применении изменений: {e}")
                    else:
                        st.warning("⚠️ Поле для изменений пусто.")

            else:
                st.error(f"Секция '{section_to_edit}' не найдена в конфигурации ассистента.")

        except Exception as e:
            st.error(f"Критическая ошибка при отображении настроек: {e}")

        st.markdown('</div>', unsafe_allow_html=True)


    def _save_interaction(self, task: str, result: str, task_type: str, duration_ms: Optional[float] = None) -> None:
        """Thread-safe сохранение взаимодействия с comprehensive метриками"""
        try:
            interaction = InteractionRecord(
                timestamp=datetime.now().isoformat(),
                task=task[:500],  # Ограничение для оптимизации памяти
                result=result[:1000],  # Ограничение для оптимизации памяти
                task_type=task_type,
                duration_ms=duration_ms,
                success=True
            )

            st.session_state.conversation_history.append(asdict(interaction))

            # Сохранение в ассистенте с проверкой доступности
            if st.session_state.assistant and hasattr(st.session_state.assistant, 'save_interaction'):
                try:
                    st.session_state.assistant.save_interaction(
                        task=task[:200],
                        prompt=task[:300],
                        result=result[:400],
                        task_type=task_type
                    )
                except Exception as e:
                    logger.warning(f"Не удалось сохранить взаимодействие в ассистенте: {e}")

            logger.info(f"Взаимодействие сохранено: {task_type}, длительность: {duration_ms}мс")

        except Exception as e:
            logger.error(f"Ошибка сохранения взаимодействия: {e}")
            # Не прерываем выполнение, только логируем

    def _get_file_extension(self, language: str) -> str:
        """Получение расширения файла по языку программирования с расширенной поддержкой"""
        extensions = {
            'python': 'py',
            'javascript': 'js',
            'typescript': 'ts',
            'java': 'java',
            'c++': 'cpp',
            'c': 'c',
            'c#': 'cs',
            'go': 'go',
            'rust': 'rs',
            'php': 'php',
            'ruby': 'rb',
            'swift': 'swift',
            'kotlin': 'kt',
            'scala': 'scala',
            'html': 'html',
            'css': 'css',
            'sql': 'sql',
            'bash': 'sh',
            'r': 'r',
            'matlab': 'm',
            'lua': 'lua'
        }
        return extensions.get(language.lower(), 'txt')

    def run(self) -> None:
        """
        Главный метод запуска приложения с comprehensive error handling

        Архитектурные принципы:
        - Graceful degradation при ошибках
        - Comprehensive monitoring и логирование
        - Оптимизация производительности через кэширование
        - Thread-safe операции
        """
        try:
            # Инициализация и проверка состояния
            current_time = time.time()

            # Throttling для предотвращения избыточных перерисовок
            if (st.session_state.loading_status == ModelStatus.LOADING.value and
                current_time - st.session_state.last_rerun_time > 0.5):
                st.session_state.last_rerun_time = current_time

            # Рендеринг основного интерфейса
            with self._error_boundary("рендеринг заголовка"):
                self.render_header()

            with self._error_boundary("рендеринг боковой панели"):
                self.render_sidebar()

            with self._error_boundary("рендеринг основного интерфейса"):
                self.render_main_interface()

            # Production-ready футер с системной информацией
            self._render_footer()

        except Exception as e:
            logger.critical(f"Критическая ошибка выполнения приложения: {e}")
            st.error("🚨 Критическая ошибка приложения")

            # Emergency состояние с базовой функциональностью
            st.markdown("""
            <div class="error-section">
                <h3>🛠️ Режим восстановления</h3>
                <p>Приложение работает в ограниченном режиме. Попробуйте:</p>
                <ul>
                    <li>Обновить страницу</li>
                    <li>Очистить кэш браузера</li>
                    <li>Проверить системные ресурсы</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # Debug информация для разработчиков
            if st.session_state.get('debug_mode', False):
                st.text_area("Critical Error Traceback:", traceback.format_exc(), height=300)

    def _render_footer(self) -> None:
        """Production-ready футер с системной информацией"""
        st.markdown("---")

        footer_col1, footer_col2, footer_col3 = st.columns(3)

        with footer_col1:
            st.markdown(
                f"🤖 **AIDA v2.0** | Session: {st.session_state.session_start.strftime('%H:%M:%S')}"
            )

        with footer_col2:
            if st.session_state.model_loaded:
                model_status = "🟢 Модель активна"
            elif st.session_state.loading_status == ModelStatus.LOADING.value:
                model_status = "🟡 Загрузка модели"
            else:
                model_status = "🔴 Модель не загружена"
            st.markdown(f"**Статус:** {model_status}")

        with footer_col3:
            metrics = self._get_system_metrics()
            st.markdown(f"**Ресурсы:** RAM {metrics.ram_percent:.0f}% | CPU {metrics.cpu_percent:.0f}%")

        # Дополнительная техническая информация
        with st.expander("🔧 Техническая информация"):
            tech_col1, tech_col2 = st.columns(2)

            with tech_col1:
                st.write("**Системные характеристики:**")
                st.write(f"- Python: {sys.version.split()[0]}")
                st.write(f"- Streamlit: {st.__version__}")
                st.write(f"- GPU доступен: {'✅' if gpu_available else '❌'}")
                st.write(f"- Процессов: {psutil.cpu_count()}")

            with tech_col2:
                st.write("**Статистика сессии:**")
                st.write(f"- Задач выполнено: {len(st.session_state.conversation_history)}")
                st.write(f"- Сообщений в чате: {len(st.session_state.chat_messages)}")
                uptime = datetime.now() - st.session_state.session_start
                st.write(f"- Время работы: {int(uptime.total_seconds() / 60)}м {int(uptime.total_seconds() % 60)}с")


# ===== ТОЧКА ВХОДА =====
def main() -> None:
    """
    Главная функция с comprehensive error handling и graceful degradation

    Production-ready архитектура:
    - Централизованная обработка ошибок
    - Логирование всех критических операций
    - Graceful fallback при отказах компонентов
    """
    try:
        # Инициализация приложения
        logger.info("Запуск Enhanced AIDA Web Interface")
        app = EnhancedAidaWebInterface()

        # Запуск основного цикла
        app.run()

    except Exception as e:
        logger.critical(f"Критическая ошибка инициализации приложения: {e}")

        # Emergency UI для критических ошибок
        st.error("🚨 Критическая ошибка инициализации")
        st.markdown("""
        <div class="error-section">
            <h3>Системная ошибка</h3>
            <p>Не удалось запустить приложение. Возможные причины:</p>
            <ul>
                <li>Отсутствуют необходимые зависимости</li>
                <li>Недостаточно системных ресурсов</li>
                <li>Проблемы с конфигурацией</li>
            </ul>
            <p><strong>Рекомендации:</strong></p>
            <ul>
                <li>Проверьте установку всех зависимостей</li>
                <li>Убедитесь в наличии файла fixed_aida_code3.py</li>
                <li>Проверьте доступность системных ресурсов</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Отображение технических деталей для разработчиков
        with st.expander("🔧 Техническая информация"):
            st.text_area("Error Details:", str(e), height=150)
            st.text_area("Full Traceback:", traceback.format_exc(), height=300)


if __name__ == "__main__": 
    main()
