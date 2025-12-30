const form = document.getElementById('loginForm');
const errorBox = document.getElementById('authError');

const showError = (message) => {
    if (!errorBox) return;
    errorBox.textContent = message;
    errorBox.hidden = false;
};

if (form) {
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        errorBox.hidden = true;

        const data = {
            email: form.email.value.trim(),
            password: form.password.value,
        };

        const submitBtn = form.querySelector('.submit-btn');
        submitBtn.disabled = true;

        try {
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            if (!response.ok) {
                const payload = await response.json().catch(() => ({}));
                throw new Error(payload.detail || 'Invalid credentials');
            }

            window.location.href = '/dashboard';
        } catch (error) {
            showError(error.message || 'Unable to sign in');
        } finally {
            submitBtn.disabled = false;
        }
    });
}
