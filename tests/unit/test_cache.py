import pytest
import asyncio
from src.core.cache import TokenCache

@pytest.mark.asyncio
async def test_token_cache_set_get_cleanup():
    cache = TokenCache(max_size=10, default_ttl=1)
    await cache.set('token1', {'price': 1.23})
    value = await cache.get('token1')
    assert value == {'price': 1.23}
    await asyncio.sleep(1.1)
    # Should expire
    value = await cache.get('token1')
    assert value is None
    # Test cleanup
    await cache.set('token2', {'price': 2.34})
    await cache.cleanup()
    value = await cache.get('token2')
    assert value == {'price': 2.34} 