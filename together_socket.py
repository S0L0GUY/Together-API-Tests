from together import Together


class TogetherClient:
    def __init__(
        self,
        model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        lm_temperature=0.7,
        conversation_history=[],
        server_ip=None,
    ):
        """
        Initializes the LMStudio wrapper but uses the Together
        client under the hood.

        Args:
            model_id (str): The identifier for the language
                model.
            lm_temperature (float): The temperature setting
                for the language model.
            conversation_history (list): The history of the
                conversation.
            server_ip (str, optional): Kept for backwards
                compatibility but unused for Together.

        Attributes:
            model_id (str): The identifier for the language
                model.
            lm_temperature (float): The temperature setting
                for the language model.
            history (list): The history of the conversation.
            server_ip (str): The server_ip passed in (kept
                for compatibility).
            client (Together): The Together client instance.
        """

        # Together client; uses environment or default config for auth.
        self.client = Together()

        self.model_id = model_id
        self.lm_temperature = lm_temperature
        self.history = conversation_history
        self.server_ip = server_ip

    def add_user_message(self, message):
        """
        Adds a user message to the history.

        Args:
            message (str): The message content to add to history.
        """
        self.history.append({"role": "user", "content": message})

    def add_assistant_message(self, message):
        """
        Adds a message from the assistant to the conversation history.

        Args:
            message (str): The message content to add to history.
        """
        self.history.append({"role": "assistant", "content": message})

    def add_system_message(self, message):
        """
        Adds a system message to the conversation history.

        Args:
            message (str): The system message to add to history.
        """
        self.history.append({"role": "system", "content": message})

    def retrieve_conversion(self):
        """
        Retrieves the conversation history.

        Returns:
            list: The conversation history.
        """
        return self.history

    def generate_response(self, live_print=False):
        """
        Generate a response from the assistant model using the
        conversation history.

        Args:
            live_print (bool, optional): If True, prints the
                response after generation. Defaults to False.

        Returns:
            str: The generated response content from the
                assistant.
        """

        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=self.history,
            temperature=self.lm_temperature,
        )

        # Defensive extraction to handle object-like or
        # dict-like responses from different Together versions.
        content = None
        try:
            choices = getattr(completion, "choices", None)
            if choices:
                first = choices[0]
                msg = getattr(first, "message", None)
                if msg is not None:
                    content = getattr(msg, "content", None)
        except Exception:
            content = None

        if content is None:
            # If the response supports mapping access, try that.
            if hasattr(completion, "__getitem__"):
                try:
                    content = completion["choices"][0]["message"]["content"]
                except Exception:
                    content = None

            if content is None:
                try:
                    content = str(completion)
                except Exception:
                    content = ""

        if live_print and content:
            print(content)

        self.add_assistant_message(content)
        return content
