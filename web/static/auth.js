const TaxAuth = (() => {
    const roles = ['user', 'admin'];
    const tokenKey = role => 'tax_agent_' + role + '_session';
    const storedToken = role => sessionStorage.getItem(tokenKey(role)) || '';
    const loginRequired = '请先登录';
    for (const role of roles) sessionStorage.removeItem('tax_agent_' + role + '_token');
    let session = null;
    let verification = 0;

    function setSession(next, message = '') {
        session?.controller.abort();
        session = next;
        window.dispatchEvent(new CustomEvent('tax-auth-changed', {detail: {message}}));
    }

    function assertCurrent(expected) {
        if (expected !== session) throw new DOMException('身份已变更', 'AbortError');
        if (!session) throw new Error(loginRequired);
    }

    async function verify(value, storedRole) {
        const attempt = ++verification;
        let token = value;
        if (!storedRole) {
            const credential = value.trim();
            if (!credential) throw new Error('请填写登录凭证');
            const login = await fetch('/api/auth/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({credential}),
                cache: 'no-store',
                redirect: 'error',
                signal: AbortSignal.timeout(15000),
            });
            const result = await login.json().catch(() => ({}));
            if (!login.ok || !result.access_token) throw new Error(result.detail || '登录失败，请稍后重试。');
            token = result.access_token;
        }
        const response = await fetch('/api/me', {
            headers: {Authorization: 'Bearer ' + token},
            cache: 'no-store',
            redirect: 'error',
            signal: AbortSignal.timeout(15000),
        });
        const user = await response.json().catch(() => ({}));
        if (attempt !== verification) throw new DOMException('验证已取消', 'AbortError');
        if (response.status === 401 && storedRole && storedToken(storedRole) === value) {
            sessionStorage.removeItem(tokenKey(storedRole));
        }
        if (!response.ok || !roles.includes(user.role) || !user.id) {
            throw new Error(typeof user.detail === 'string' ? user.detail : '无法验证登录状态，请稍后重试。');
        }
        if (storedRole && storedRole !== user.role && storedToken(storedRole) === value) {
            sessionStorage.removeItem(tokenKey(storedRole));
        }
        sessionStorage.setItem(tokenKey(user.role), token);
        sessionStorage.setItem('tax_agent_role', user.role);
        setSession({user, token, controller: new AbortController()});
        return user;
    }

    async function useStored(role, optional = false) {
        const token = storedToken(role);
        if (!token && optional) return null;
        return verify(token, role);
    }

    function restore() {
        const role = sessionStorage.getItem('tax_agent_role');
        return useStored(roles.includes(role) ? role : 'user', true);
    }

    async function checkResponse(status, readError, expected) {
        assertCurrent(expected);
        if (status !== 401) return;
        const data = await readError().catch(() => ({}));
        assertCurrent(expected);
        const message = typeof data.detail === 'string' ? data.detail : '访问 Token 已失效，请重新登录';
        if (storedToken(expected.user.role) === expected.token) {
            sessionStorage.removeItem(tokenKey(expected.user.role));
        }
        setSession(null, message);
        throw new DOMException(message, 'AbortError');
    }

    function authorization(expected) {
        assertCurrent(expected);
        return 'Bearer ' + expected.token;
    }

    async function request(path, options = {}) {
        const expected = session;
        const headers = new Headers(options.headers);
        headers.set('Authorization', authorization(expected));
        const sessionSignal = expected.controller.signal;
        const signal = options.signal ? AbortSignal.any([options.signal, sessionSignal]) : sessionSignal;
        const response = await fetch(path, {...options, headers, signal, cache: 'no-store', redirect: 'error'});
        await checkResponse(response.status, () => response.json(), expected);
        return response;
    }

    return {verify, useStored, restore, hasStored: role => !!storedToken(role), request, authorization, checkResponse,
        snapshot: () => session, user: () => session?.user ?? null};
})();

window.addEventListener('pageshow', event => {
    if (event.persisted) window.location.reload();
});
