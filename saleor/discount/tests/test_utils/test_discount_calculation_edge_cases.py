"""Edge-case tests for discount calculation utilities.

Covers: expired vouchers, stacked promotions, boundary values,
voucher lifecycle helpers, and manual-discount splitting.
"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from django.utils import timezone
from prices import Money

from ... import (
    DiscountType,
    DiscountValueType,
    PromotionRuleInfo,
    RewardValueType,
    VoucherType,
)
from ...models import (
    NotApplicable,
    Promotion,
    PromotionRule,
    Voucher,
    VoucherChannelListing,
    VoucherCode,
)
from ...utils.manual_discount import apply_discount_to_value
from ...utils.promotion import (
    calculate_discounted_price_for_rules,
    get_best_promotion_discount,
    get_product_discount_on_promotion,
    get_sale_id,
    is_discounted_line_by_catalogue_promotion,
    prepare_promotion_discount_reason,
)
from ...utils.voucher import (
    get_products_voucher_discount,
    get_voucher_code_instance,
    increase_voucher_usage,
    is_line_level_voucher,
    is_shipping_voucher,
    release_voucher_code_usage,
    validate_voucher,
)

# ---------------------------------------------------------------------------
# Expired / inactive voucher edge cases
# ---------------------------------------------------------------------------


def test_expired_voucher_not_active(channel_USD):
    """Voucher whose end_date is in the past is not returned by active_in_channel."""
    # given
    voucher = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.FIXED,
        start_date=timezone.now() - datetime.timedelta(days=10),
        end_date=timezone.now() - datetime.timedelta(days=1),
    )
    VoucherCode.objects.create(code="expired-code", voucher=voucher)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount=Money(10, channel_USD.currency_code),
    )

    # when
    active_vouchers = Voucher.objects.active_in_channel(
        date=timezone.now(), channel_slug=channel_USD.slug
    )

    # then
    assert voucher not in active_vouchers


def test_expired_voucher_code_instance_raises(channel_USD):
    """get_voucher_code_instance raises InvalidPromoCode for expired voucher."""
    # given
    from ....core.utils.promo_code import InvalidPromoCode

    voucher = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.FIXED,
        start_date=timezone.now() - datetime.timedelta(days=10),
        end_date=timezone.now() - datetime.timedelta(days=1),
    )
    code = "expired-promo"
    VoucherCode.objects.create(code=code, voucher=voucher)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount=Money(10, channel_USD.currency_code),
    )

    # when & then
    with pytest.raises(InvalidPromoCode):
        get_voucher_code_instance(code, channel_USD.slug)


def test_voucher_not_yet_started(channel_USD):
    """Voucher whose start_date is in the future is not active."""
    # given
    voucher = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.FIXED,
        start_date=timezone.now() + datetime.timedelta(days=5),
    )
    VoucherCode.objects.create(code="future-code", voucher=voucher)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount=Money(10, channel_USD.currency_code),
    )

    # when
    active_vouchers = Voucher.objects.active_in_channel(
        date=timezone.now(), channel_slug=channel_USD.slug
    )

    # then
    assert voucher not in active_vouchers


def test_deactivated_voucher_code_raises(channel_USD):
    """get_voucher_code_instance raises for deactivated code (single-use spent)."""
    # given
    from ....core.utils.promo_code import InvalidPromoCode

    voucher = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.FIXED,
        single_use=True,
    )
    code_instance = VoucherCode.objects.create(
        code="single-used", voucher=voucher, is_active=False
    )
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount=Money(10, channel_USD.currency_code),
    )

    # when & then
    with pytest.raises(InvalidPromoCode):
        get_voucher_code_instance(code_instance.code, channel_USD.slug)


def test_usage_limit_reached_voucher(channel_USD):
    """Voucher with all codes at usage limit is not active."""
    # given
    voucher = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.FIXED,
        usage_limit=5,
    )
    VoucherCode.objects.create(code="maxed-out", voucher=voucher, used=5)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount=Money(10, channel_USD.currency_code),
    )

    # when
    active = Voucher.objects.active_in_channel(
        date=timezone.now(), channel_slug=channel_USD.slug
    )

    # then
    assert voucher not in active


# ---------------------------------------------------------------------------
# Voucher lifecycle helpers
# ---------------------------------------------------------------------------


def test_increase_voucher_usage_all_flags(channel_USD, customer_user):
    """increase_voucher_usage respects usage_limit, once-per-customer, single_use."""
    # given
    voucher = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.FIXED,
        usage_limit=100,
        apply_once_per_customer=True,
        single_use=True,
    )
    code_instance = VoucherCode.objects.create(code="all-flags", voucher=voucher)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount=Money(10, channel_USD.currency_code),
    )

    # when
    increase_voucher_usage(voucher, code_instance, customer_user.email)

    # then
    code_instance.refresh_from_db()
    assert code_instance.used == 1
    assert code_instance.is_active is False


def test_increase_voucher_usage_no_usage_limit(channel_USD):
    """When usage_limit is None, code.used is NOT incremented."""
    # given
    voucher = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.FIXED,
        usage_limit=None,
        single_use=True,
    )
    code_instance = VoucherCode.objects.create(code="no-limit", voucher=voucher)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount=Money(10, channel_USD.currency_code),
    )

    # when
    increase_voucher_usage(voucher, code_instance, "user@example.com")

    # then
    code_instance.refresh_from_db()
    assert code_instance.used == 0
    assert code_instance.is_active is False


def test_release_voucher_code_usage_no_code():
    """release_voucher_code_usage returns early when code is None."""
    # when & then — should not raise
    release_voucher_code_usage(None, None, "user@example.com")


def test_release_voucher_code_usage_all_fields(channel_USD):
    """release_voucher_code_usage re-activates code and decreases usage."""
    # given
    voucher = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.FIXED,
        usage_limit=100,
        single_use=True,
    )
    code_instance = VoucherCode.objects.create(
        code="release-test", voucher=voucher, used=3, is_active=False
    )
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount=Money(10, channel_USD.currency_code),
    )

    # when
    release_voucher_code_usage(code_instance, voucher, None)

    # then
    code_instance.refresh_from_db()
    assert code_instance.used == 2
    assert code_instance.is_active is True


def test_release_voucher_code_usage_with_email(channel_USD, customer_user):
    """release_voucher_code_usage removes VoucherCustomer record."""
    # given
    from ...models import VoucherCustomer

    voucher = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.FIXED,
    )
    code_instance = VoucherCode.objects.create(code="release-email", voucher=voucher)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount=Money(10, channel_USD.currency_code),
    )
    VoucherCustomer.objects.create(
        voucher_code=code_instance, customer_email=customer_user.email
    )

    # when
    release_voucher_code_usage(code_instance, voucher, customer_user.email)

    # then
    assert not VoucherCustomer.objects.filter(
        voucher_code=code_instance, customer_email=customer_user.email
    ).exists()


# ---------------------------------------------------------------------------
# Voucher type helpers
# ---------------------------------------------------------------------------


def test_is_shipping_voucher_true():
    # given
    voucher = Voucher(type=VoucherType.SHIPPING)

    # when & then
    assert is_shipping_voucher(voucher) is True


def test_is_shipping_voucher_false():
    # given
    voucher = Voucher(type=VoucherType.ENTIRE_ORDER)

    # when & then
    assert is_shipping_voucher(voucher) is False


def test_is_shipping_voucher_none():
    # when & then
    assert is_shipping_voucher(None) is False


def test_is_line_level_voucher_specific_product():
    # given
    voucher = Voucher(type=VoucherType.SPECIFIC_PRODUCT)

    # when & then
    assert is_line_level_voucher(voucher)


def test_is_line_level_voucher_apply_once_per_order():
    # given
    voucher = Voucher(type=VoucherType.ENTIRE_ORDER, apply_once_per_order=True)

    # when & then
    assert is_line_level_voucher(voucher)


def test_is_line_level_voucher_entire_order_no_once():
    # given
    voucher = Voucher(type=VoucherType.ENTIRE_ORDER, apply_once_per_order=False)

    # when & then
    assert not is_line_level_voucher(voucher)


def test_is_line_level_voucher_none():
    # when & then
    assert not is_line_level_voucher(None)


# ---------------------------------------------------------------------------
# Boundary values — discount calculations
# ---------------------------------------------------------------------------


def test_fixed_discount_exceeds_price(channel_USD):
    """Fixed discount larger than price should not go below zero."""
    # given
    voucher = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.FIXED,
    )
    VoucherCode.objects.create(code="big-discount", voucher=voucher)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount=Money(999, channel_USD.currency_code),
    )
    price = Money(10, "USD")

    # when
    discount_amount = voucher.get_discount_amount_for(price, channel_USD)

    # then
    assert discount_amount == price


def test_percentage_discount_100(channel_USD):
    """100% discount should return the full price as discount amount."""
    # given
    voucher = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.PERCENTAGE,
    )
    VoucherCode.objects.create(code="full-off", voucher=voucher)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount_value=Decimal(100),
        currency=channel_USD.currency_code,
    )
    price = Money(50, "USD")

    # when
    discount_amount = voucher.get_discount_amount_for(price, channel_USD)

    # then
    assert discount_amount == price


def test_percentage_discount_zero(channel_USD):
    """0% discount should return zero discount amount."""
    # given
    voucher = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.PERCENTAGE,
    )
    VoucherCode.objects.create(code="zero-pct", voucher=voucher)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount_value=Decimal(0),
        currency=channel_USD.currency_code,
    )
    price = Money(50, "USD")

    # when
    discount_amount = voucher.get_discount_amount_for(price, channel_USD)

    # then
    assert discount_amount == Money(0, "USD")


def test_fixed_discount_zero_price(channel_USD):
    """Discount on a zero-priced item should yield zero discount."""
    # given
    voucher = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.FIXED,
    )
    VoucherCode.objects.create(code="zero-price", voucher=voucher)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount=Money(10, channel_USD.currency_code),
    )
    price = Money(0, "USD")

    # when
    discount_amount = voucher.get_discount_amount_for(price, channel_USD)

    # then
    assert discount_amount == Money(0, "USD")


def test_voucher_not_assigned_to_channel_raises(channel_USD, channel_PLN):
    """get_discount raises NotApplicable when voucher lacks channel listing."""
    # given
    voucher = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.FIXED,
    )
    VoucherCode.objects.create(code="wrong-channel", voucher=voucher)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount=Money(10, channel_USD.currency_code),
    )

    # when & then
    with pytest.raises(NotApplicable):
        voucher.get_discount(channel_PLN)


# ---------------------------------------------------------------------------
# Products voucher discount
# ---------------------------------------------------------------------------


def test_get_products_voucher_discount_apply_once_per_order(channel_USD):
    """apply_once_per_order: discount applied only on cheapest product."""
    # given
    voucher = Voucher.objects.create(
        type=VoucherType.SPECIFIC_PRODUCT,
        discount_value_type=DiscountValueType.FIXED,
        apply_once_per_order=True,
    )
    VoucherCode.objects.create(code="once-per", voucher=voucher)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount=Money(5, channel_USD.currency_code),
    )
    prices = [Money(10, "USD"), Money(20, "USD"), Money(30, "USD")]

    # when
    discount = get_products_voucher_discount(voucher, prices, channel_USD)

    # then — discount applied only to cheapest (10) → 5
    assert discount == Money(5, "USD")


def test_get_products_voucher_discount_all_items(channel_USD):
    """Without apply_once_per_order, discount applied to each item."""
    # given
    voucher = Voucher.objects.create(
        type=VoucherType.SPECIFIC_PRODUCT,
        discount_value_type=DiscountValueType.FIXED,
        apply_once_per_order=False,
    )
    VoucherCode.objects.create(code="all-items", voucher=voucher)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount=Money(3, channel_USD.currency_code),
    )
    prices = [Money(10, "USD"), Money(20, "USD")]

    # when
    discount = get_products_voucher_discount(voucher, prices, channel_USD)

    # then — 3 + 3 = 6
    assert discount == Money(6, "USD")


def test_get_products_voucher_discount_exceeds_all_prices(channel_USD):
    """Fixed discount larger than each product price should cap per-item."""
    # given
    voucher = Voucher.objects.create(
        type=VoucherType.SPECIFIC_PRODUCT,
        discount_value_type=DiscountValueType.FIXED,
        apply_once_per_order=False,
    )
    VoucherCode.objects.create(code="big-fixed", voucher=voucher)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount=Money(100, channel_USD.currency_code),
    )
    prices = [Money(5, "USD"), Money(8, "USD")]

    # when
    discount = get_products_voucher_discount(voucher, prices, channel_USD)

    # then — each item is fully discounted: 5 + 8 = 13
    assert discount == Money(13, "USD")


# ---------------------------------------------------------------------------
# Validate voucher — edge cases
# ---------------------------------------------------------------------------


def test_validate_voucher_exact_min_spent_boundary(channel_USD):
    """Voucher with min_spent equal to total price should pass."""
    # given
    min_spent = Decimal(50)
    voucher = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.FIXED,
    )
    VoucherCode.objects.create(code="boundary", voucher=voucher)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount=Money(10, channel_USD.currency_code),
        min_spent_amount=min_spent,
    )
    total_price = Money(min_spent, "USD")

    # when & then — should not raise
    validate_voucher(voucher, total_price, 1, "test@example.com", channel_USD, None)


def test_validate_voucher_one_cent_below_min_spent(channel_USD):
    """One cent below min_spent should raise NotApplicable."""
    # given
    min_spent = Decimal(50)
    voucher = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.FIXED,
    )
    VoucherCode.objects.create(code="one-cent", voucher=voucher)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount=Money(10, channel_USD.currency_code),
        min_spent_amount=min_spent,
    )
    total_price = Money(Decimal("49.99"), "USD")

    # when & then
    with pytest.raises(NotApplicable):
        validate_voucher(voucher, total_price, 1, "test@example.com", channel_USD, None)


def test_validate_voucher_exact_min_quantity_boundary(channel_USD):
    """Voucher with min_checkout_items_quantity equal to quantity should pass."""
    # given
    min_qty = 3
    voucher = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.FIXED,
        min_checkout_items_quantity=min_qty,
    )
    VoucherCode.objects.create(code="qty-boundary", voucher=voucher)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount=Money(10, channel_USD.currency_code),
    )
    total_price = Money(100, "USD")

    # when & then — should not raise
    validate_voucher(
        voucher, total_price, min_qty, "test@example.com", channel_USD, None
    )


def test_validate_voucher_one_below_min_quantity(channel_USD):
    """One item below min_checkout_items_quantity should raise."""
    # given
    min_qty = 3
    voucher = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.FIXED,
        min_checkout_items_quantity=min_qty,
    )
    VoucherCode.objects.create(code="qty-below", voucher=voucher)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount=Money(10, channel_USD.currency_code),
    )
    total_price = Money(100, "USD")

    # when & then
    with pytest.raises(NotApplicable):
        validate_voucher(
            voucher, total_price, min_qty - 1, "test@example.com", channel_USD, None
        )


# ---------------------------------------------------------------------------
# Stacked promotions — calculate_discounted_price_for_rules
# ---------------------------------------------------------------------------


def test_stacked_rules_summed(channel_USD):
    """Multiple rules discount amounts are summed and applied to the price."""
    # given
    promotion = Promotion.objects.create(name="Stack promo")
    rule_10pct = PromotionRule.objects.create(
        promotion=promotion,
        reward_value_type=RewardValueType.PERCENTAGE,
        reward_value=Decimal(10),
    )
    rule_5fix = PromotionRule.objects.create(
        promotion=promotion,
        reward_value_type=RewardValueType.FIXED,
        reward_value=Decimal(5),
    )
    currency = channel_USD.currency_code
    price = Money(100, currency)

    # when
    result = calculate_discounted_price_for_rules(
        price=price, rules=[rule_10pct, rule_5fix], currency=currency
    )

    # then — 10% of 100 = 10, fixed 5 → total discount 15, result 85
    assert result == Money(85, currency)


def test_stacked_rules_cannot_go_below_zero(channel_USD):
    """When stacked rules exceed the price, result is clamped to zero."""
    # given
    promotion = Promotion.objects.create(name="Over-discount")
    rule_80pct = PromotionRule.objects.create(
        promotion=promotion,
        reward_value_type=RewardValueType.PERCENTAGE,
        reward_value=Decimal(80),
    )
    rule_big_fix = PromotionRule.objects.create(
        promotion=promotion,
        reward_value_type=RewardValueType.FIXED,
        reward_value=Decimal(50),
    )
    currency = channel_USD.currency_code
    price = Money(100, currency)

    # when
    result = calculate_discounted_price_for_rules(
        price=price, rules=[rule_80pct, rule_big_fix], currency=currency
    )

    # then — 80 + 50 = 130 discount > 100, clamped to 0
    assert result == Money(0, currency)


def test_stacked_rules_empty_list(channel_USD):
    """No rules should return the original price."""
    # given
    currency = channel_USD.currency_code
    price = Money(100, currency)

    # when
    result = calculate_discounted_price_for_rules(
        price=price, rules=[], currency=currency
    )

    # then
    assert result == price


def test_single_percentage_rule(channel_USD):
    """Single percentage rule applied correctly."""
    # given
    promotion = Promotion.objects.create(name="Single pct")
    rule = PromotionRule.objects.create(
        promotion=promotion,
        reward_value_type=RewardValueType.PERCENTAGE,
        reward_value=Decimal(25),
    )
    currency = channel_USD.currency_code
    price = Money(200, currency)

    # when
    result = calculate_discounted_price_for_rules(
        price=price, rules=[rule], currency=currency
    )

    # then
    assert result == Money(150, currency)


def test_single_fixed_rule(channel_USD):
    """Single fixed rule applied correctly."""
    # given
    promotion = Promotion.objects.create(name="Single fix")
    rule = PromotionRule.objects.create(
        promotion=promotion,
        reward_value_type=RewardValueType.FIXED,
        reward_value=Decimal(30),
    )
    currency = channel_USD.currency_code
    price = Money(100, currency)

    # when
    result = calculate_discounted_price_for_rules(
        price=price, rules=[rule], currency=currency
    )

    # then
    assert result == Money(70, currency)


# ---------------------------------------------------------------------------
# get_best_promotion_discount
# ---------------------------------------------------------------------------


def test_best_promotion_discount_picks_highest(channel_USD):
    """get_best_promotion_discount returns the rule with the largest discount."""
    # given
    promotion = Promotion.objects.create(name="Best pick")
    rule_small = PromotionRule.objects.create(
        promotion=promotion,
        reward_value_type=RewardValueType.FIXED,
        reward_value=Decimal(5),
    )
    rule_big = PromotionRule.objects.create(
        promotion=promotion,
        reward_value_type=RewardValueType.FIXED,
        reward_value=Decimal(20),
    )
    rule_small.channels.add(channel_USD)
    rule_big.channels.add(channel_USD)

    rules_info = [
        PromotionRuleInfo(rule=rule_small, channel_ids=[channel_USD.id]),
        PromotionRuleInfo(rule=rule_big, channel_ids=[channel_USD.id]),
    ]
    price = Money(100, channel_USD.currency_code)

    # when
    result = get_best_promotion_discount(price, rules_info, channel_USD)

    # then
    assert result is not None
    rule_id, discount_amount = result
    assert rule_id == rule_big.id
    assert discount_amount == Money(20, channel_USD.currency_code)


def test_best_promotion_discount_no_matching_channel(channel_USD, channel_PLN):
    """No discount returned when rules don't match the channel."""
    # given
    promotion = Promotion.objects.create(name="Wrong channel")
    rule = PromotionRule.objects.create(
        promotion=promotion,
        reward_value_type=RewardValueType.FIXED,
        reward_value=Decimal(10),
    )
    rule.channels.add(channel_PLN)

    rules_info = [
        PromotionRuleInfo(rule=rule, channel_ids=[channel_PLN.id]),
    ]
    price = Money(100, channel_USD.currency_code)

    # when
    result = get_best_promotion_discount(price, rules_info, channel_USD)

    # then
    assert result is None


# ---------------------------------------------------------------------------
# get_product_discount_on_promotion
# ---------------------------------------------------------------------------


def test_get_product_discount_on_promotion_wrong_channel(channel_USD, channel_PLN):
    """Raises NotApplicable when rule's channel doesn't match."""
    # given
    promotion = Promotion.objects.create(name="Wrong ch")
    rule = PromotionRule.objects.create(
        promotion=promotion,
        reward_value_type=RewardValueType.FIXED,
        reward_value=Decimal(10),
    )
    rule_info = PromotionRuleInfo(rule=rule, channel_ids=[channel_PLN.id])

    # when & then
    with pytest.raises(NotApplicable):
        get_product_discount_on_promotion(rule_info, channel_USD)


def test_get_product_discount_on_promotion_matching_channel(channel_USD):
    """Returns rule id and discount callable when channel matches."""
    # given
    promotion = Promotion.objects.create(name="Match ch")
    rule = PromotionRule.objects.create(
        promotion=promotion,
        reward_value_type=RewardValueType.FIXED,
        reward_value=Decimal(10),
    )
    rule_info = PromotionRuleInfo(rule=rule, channel_ids=[channel_USD.id])

    # when
    rule_id, discount_fn = get_product_discount_on_promotion(rule_info, channel_USD)

    # then
    assert rule_id == rule.id
    price = Money(100, channel_USD.currency_code)
    assert discount_fn(price) == Money(90, channel_USD.currency_code)


# ---------------------------------------------------------------------------
# is_discounted_line_by_catalogue_promotion
# ---------------------------------------------------------------------------


def test_is_discounted_line_price_equals_discounted():
    """When price equals discounted_price, line is NOT discounted."""
    # given
    listing = MagicMock()
    listing.price_amount = Decimal(100)
    listing.discounted_price_amount = Decimal(100)

    # when & then
    assert is_discounted_line_by_catalogue_promotion(listing) is False


def test_is_discounted_line_price_none():
    """When price_amount is None, line is NOT discounted."""
    # given
    listing = MagicMock()
    listing.price_amount = None
    listing.discounted_price_amount = Decimal(90)

    # when & then
    assert is_discounted_line_by_catalogue_promotion(listing) is False


def test_is_discounted_line_discounted_price_none():
    """When discounted_price_amount is None, line is NOT discounted."""
    # given
    listing = MagicMock()
    listing.price_amount = Decimal(100)
    listing.discounted_price_amount = None

    # when & then
    assert is_discounted_line_by_catalogue_promotion(listing) is False


def test_is_discounted_line_true():
    """When price differs from discounted price, line IS discounted."""
    # given
    listing = MagicMock()
    listing.price_amount = Decimal(100)
    listing.discounted_price_amount = Decimal(80)

    # when & then
    assert is_discounted_line_by_catalogue_promotion(listing) is True


# ---------------------------------------------------------------------------
# apply_discount_to_value edge cases
# ---------------------------------------------------------------------------


def test_apply_discount_to_value_percentage():
    """Percentage discount should reduce money by the given percent."""
    # given
    price = Money(200, "USD")

    # when
    result = apply_discount_to_value(
        value=Decimal(25),
        value_type=DiscountValueType.PERCENTAGE,
        currency="USD",
        price_to_discount=price,
    )

    # then
    assert result == Money(150, "USD")


def test_apply_discount_to_value_fixed():
    """Fixed discount should subtract the value from the price."""
    # given
    price = Money(200, "USD")

    # when
    result = apply_discount_to_value(
        value=Decimal(50),
        value_type=DiscountValueType.FIXED,
        currency="USD",
        price_to_discount=price,
    )

    # then
    assert result == Money(150, "USD")


def test_apply_discount_to_value_fixed_exceeds_price():
    """Fixed discount exceeding the price is clamped to zero by `prices`."""
    # given
    price = Money(10, "USD")

    # when
    result = apply_discount_to_value(
        value=Decimal(50),
        value_type=DiscountValueType.FIXED,
        currency="USD",
        price_to_discount=price,
    )

    # then — `prices.fixed_discount` clamps at zero
    assert result.amount == Decimal(0)


def test_apply_discount_to_value_percentage_100():
    """100% discount zeroes out the price."""
    # given
    price = Money(100, "USD")

    # when
    result = apply_discount_to_value(
        value=Decimal(100),
        value_type=DiscountValueType.PERCENTAGE,
        currency="USD",
        price_to_discount=price,
    )

    # then
    assert result == Money(0, "USD")


# ---------------------------------------------------------------------------
# Promotion helpers: prepare_promotion_discount_reason, get_sale_id
# ---------------------------------------------------------------------------


def test_prepare_promotion_discount_reason_with_old_sale_id():
    """Returns 'Sale: ...' when old_sale_id is set."""
    # given
    import graphene

    promotion = Promotion(id=1, old_sale_id=42)
    expected_gid = graphene.Node.to_global_id("Sale", 42)

    # when
    reason = prepare_promotion_discount_reason(promotion)

    # then
    assert reason == f"Sale: {expected_gid}"


def test_prepare_promotion_discount_reason_without_old_sale_id():
    """Returns 'Promotion: ...' when old_sale_id is not set."""
    # given
    promotion = Promotion(id=7, old_sale_id=None)

    # when
    reason = prepare_promotion_discount_reason(promotion)

    # then
    assert reason.startswith("Promotion:")


def test_get_sale_id_with_old_sale():
    """get_sale_id returns Sale global id when old_sale_id present."""
    # given
    import graphene

    promotion = Promotion(id=1, old_sale_id=42)
    expected = graphene.Node.to_global_id("Sale", 42)

    # when
    sale_id = get_sale_id(promotion)

    # then
    assert sale_id == expected


def test_get_sale_id_without_old_sale():
    """get_sale_id returns Promotion global id when no old_sale_id."""
    # given
    import graphene

    promotion = Promotion(id=7, old_sale_id=None)
    expected = graphene.Node.to_global_id("Promotion", 7)

    # when
    sale_id = get_sale_id(promotion)

    # then
    assert sale_id == expected


# ---------------------------------------------------------------------------
# PromotionRule.get_discount edge cases
# ---------------------------------------------------------------------------


def test_promotion_rule_get_discount_fixed(channel_USD):
    """PromotionRule fixed discount returns a callable that subtracts amount."""
    # given
    promotion = Promotion.objects.create(name="Rule fix")
    rule = PromotionRule.objects.create(
        promotion=promotion,
        reward_value_type=RewardValueType.FIXED,
        reward_value=Decimal(15),
    )
    currency = channel_USD.currency_code
    price = Money(100, currency)

    # when
    discount_fn = rule.get_discount(currency)

    # then
    assert discount_fn(price) == Money(85, currency)


def test_promotion_rule_get_discount_percentage(channel_USD):
    """PromotionRule percentage discount returns correct callable."""
    # given
    promotion = Promotion.objects.create(name="Rule pct")
    rule = PromotionRule.objects.create(
        promotion=promotion,
        reward_value_type=RewardValueType.PERCENTAGE,
        reward_value=Decimal(50),
    )
    currency = channel_USD.currency_code
    price = Money(80, currency)

    # when
    discount_fn = rule.get_discount(currency)

    # then
    assert discount_fn(price) == Money(40, currency)


# ---------------------------------------------------------------------------
# Expired voucher queryset
# ---------------------------------------------------------------------------


def test_expired_queryset(channel_USD):
    """Voucher.objects.expired returns vouchers past end_date."""
    # given
    now = timezone.now()
    expired = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.FIXED,
        start_date=now - datetime.timedelta(days=10),
        end_date=now - datetime.timedelta(days=1),
    )
    VoucherCode.objects.create(code="exp-qs", voucher=expired)
    VoucherChannelListing.objects.create(
        voucher=expired,
        channel=channel_USD,
        discount=Money(10, channel_USD.currency_code),
    )

    active = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.FIXED,
        start_date=now - datetime.timedelta(days=5),
        end_date=now + datetime.timedelta(days=10),
    )
    VoucherCode.objects.create(code="act-qs", voucher=active)
    VoucherChannelListing.objects.create(
        voucher=active,
        channel=channel_USD,
        discount=Money(10, channel_USD.currency_code),
    )

    # when
    expired_qs = Voucher.objects.expired(date=now)

    # then
    assert expired in expired_qs
    assert active not in expired_qs


def test_expired_queryset_usage_limit_reached(channel_USD):
    """Voucher whose usage_limit == total used is in expired queryset."""
    # given
    now = timezone.now()
    voucher = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.FIXED,
        start_date=now - datetime.timedelta(days=5),
        usage_limit=3,
    )
    VoucherCode.objects.create(code="used-up", voucher=voucher, used=3)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount=Money(10, channel_USD.currency_code),
    )

    # when
    expired_qs = Voucher.objects.expired(date=now)

    # then
    assert voucher in expired_qs


# ---------------------------------------------------------------------------
# calculate_order_line_discount_amount_from_denormalized_voucher
# ---------------------------------------------------------------------------


def test_denormalized_voucher_percentage_multi_qty():
    """Percentage denormalized voucher on a multi-quantity line."""
    # given
    from ...utils.voucher import (
        VoucherDenormalizedInfo,
        calculate_order_line_discount_amount_from_denormalized_voucher,
    )

    total_price = Money(100, "USD")
    voucher_info = VoucherDenormalizedInfo(
        discount_value=Decimal(25),
        discount_value_type=DiscountValueType.PERCENTAGE,
        voucher_type=VoucherType.SPECIFIC_PRODUCT,
        reason=None,
        name="Test voucher",
        apply_once_per_order=False,
        origin_line_ids=[],
    )

    line = MagicMock()
    line.quantity = 2

    line_info = MagicMock()
    line_info.voucher_denormalized_info = voucher_info
    line_info.line = line

    # when
    discount = calculate_order_line_discount_amount_from_denormalized_voucher(
        line_info, total_price
    )

    # then — 25% of 100 = 25
    assert discount == Money(25, "USD")


def test_denormalized_voucher_fixed_multi_qty():
    """Fixed denormalized voucher applied per unit on a multi-quantity line."""
    # given
    from ...utils.voucher import (
        VoucherDenormalizedInfo,
        calculate_order_line_discount_amount_from_denormalized_voucher,
    )

    total_price = Money(100, "USD")
    voucher_info = VoucherDenormalizedInfo(
        discount_value=Decimal(15),
        discount_value_type=DiscountValueType.FIXED,
        voucher_type=VoucherType.SPECIFIC_PRODUCT,
        reason=None,
        name="Fixed voucher",
        apply_once_per_order=False,
        origin_line_ids=[],
    )

    line = MagicMock()
    line.quantity = 4

    line_info = MagicMock()
    line_info.voucher_denormalized_info = voucher_info
    line_info.line = line

    # when
    discount = calculate_order_line_discount_amount_from_denormalized_voucher(
        line_info, total_price
    )

    # then — unit_price = 100/4 = 25, discount per unit = 15, total = 15*4 = 60
    assert discount == Money(60, "USD")


def test_denormalized_voucher_fixed_exceeds_unit_price():
    """Fixed denormalized voucher exceeding unit price is capped."""
    # given
    from ...utils.voucher import (
        VoucherDenormalizedInfo,
        calculate_order_line_discount_amount_from_denormalized_voucher,
    )

    total_price = Money(20, "USD")
    voucher_info = VoucherDenormalizedInfo(
        discount_value=Decimal(999),
        discount_value_type=DiscountValueType.FIXED,
        voucher_type=VoucherType.SPECIFIC_PRODUCT,
        reason=None,
        name="Big fixed",
        apply_once_per_order=False,
        origin_line_ids=[],
    )

    line = MagicMock()
    line.quantity = 2

    line_info = MagicMock()
    line_info.voucher_denormalized_info = voucher_info
    line_info.line = line

    # when
    discount = calculate_order_line_discount_amount_from_denormalized_voucher(
        line_info, total_price
    )

    # then — capped at total_price
    assert discount == total_price


def test_denormalized_voucher_once_per_order():
    """apply_once_per_order denormalized voucher only discounts a single unit."""
    # given
    from ...utils.voucher import (
        VoucherDenormalizedInfo,
        calculate_order_line_discount_amount_from_denormalized_voucher,
    )

    total_price = Money(90, "USD")
    voucher_info = VoucherDenormalizedInfo(
        discount_value=Decimal(10),
        discount_value_type=DiscountValueType.FIXED,
        voucher_type=VoucherType.SPECIFIC_PRODUCT,
        reason=None,
        name="Once per order",
        apply_once_per_order=True,
        origin_line_ids=[],
    )

    line = MagicMock()
    line.quantity = 3

    line_info = MagicMock()
    line_info.voucher_denormalized_info = voucher_info
    line_info.line = line

    # when
    discount = calculate_order_line_discount_amount_from_denormalized_voucher(
        line_info, total_price
    )

    # then — unit_price = 90/3 = 30, discount = min(10, 30) = 10
    assert discount == Money(10, "USD")


def test_denormalized_voucher_once_per_order_percentage():
    """Percentage once-per-order denormalized voucher discounts only a single unit."""
    # given
    from ...utils.voucher import (
        VoucherDenormalizedInfo,
        calculate_order_line_discount_amount_from_denormalized_voucher,
    )

    total_price = Money(60, "USD")
    voucher_info = VoucherDenormalizedInfo(
        discount_value=Decimal(50),
        discount_value_type=DiscountValueType.PERCENTAGE,
        voucher_type=VoucherType.SPECIFIC_PRODUCT,
        reason=None,
        name="50% once",
        apply_once_per_order=True,
        origin_line_ids=[],
    )

    line = MagicMock()
    line.quantity = 3

    line_info = MagicMock()
    line_info.voucher_denormalized_info = voucher_info
    line_info.line = line

    # when
    discount = calculate_order_line_discount_amount_from_denormalized_voucher(
        line_info, total_price
    )

    # then — unit_price = 60/3 = 20, 50% of 20 = 10, discount = min(10, 20) = 10
    assert discount == Money(10, "USD")


def test_denormalized_voucher_no_info():
    """Returns zero when voucher_denormalized_info is None."""
    # given
    from ...utils.voucher import (
        calculate_order_line_discount_amount_from_denormalized_voucher,
    )

    total_price = Money(100, "USD")
    line_info = MagicMock()
    line_info.voucher_denormalized_info = None

    # when
    discount = calculate_order_line_discount_amount_from_denormalized_voucher(
        line_info, total_price
    )

    # then
    assert discount == Money(0, "USD")


# ---------------------------------------------------------------------------
# is_order_level_discount (shared.py)
# ---------------------------------------------------------------------------


def test_is_order_level_discount_manual_type():
    """MANUAL type discount is order-level."""
    # given
    from ...utils.shared import is_order_level_discount

    discount = MagicMock(spec=["type", "voucher"])
    discount.type = DiscountType.MANUAL
    discount.voucher = None

    # when & then
    assert is_order_level_discount(discount) is True


def test_is_order_level_discount_order_promotion_type():
    """ORDER_PROMOTION type discount is order-level."""
    # given
    from ...utils.shared import is_order_level_discount

    discount = MagicMock(spec=["type", "voucher"])
    discount.type = DiscountType.ORDER_PROMOTION
    discount.voucher = None

    # when & then
    assert is_order_level_discount(discount) is True


def test_is_order_level_discount_voucher_entire_order():
    """Voucher-type discount with ENTIRE_ORDER voucher is order-level."""
    # given
    from ...utils.shared import is_order_level_discount

    voucher = Voucher(type=VoucherType.ENTIRE_ORDER, apply_once_per_order=False)
    discount = MagicMock(spec=["type", "voucher"])
    discount.type = DiscountType.VOUCHER
    discount.voucher = voucher

    # when & then
    assert is_order_level_discount(discount) is True


def test_is_order_level_discount_voucher_specific_product():
    """Voucher-type discount with SPECIFIC_PRODUCT voucher is NOT order-level."""
    # given
    from ...utils.shared import is_order_level_discount

    voucher = Voucher(type=VoucherType.SPECIFIC_PRODUCT)
    discount = MagicMock(spec=["type", "voucher"])
    discount.type = DiscountType.VOUCHER
    discount.voucher = voucher

    # when & then
    assert is_order_level_discount(discount) is False


# ---------------------------------------------------------------------------
# Voucher.get_discount edge cases
# ---------------------------------------------------------------------------


def test_voucher_get_discount_unknown_type_raises(channel_USD):
    """Voucher with unknown discount_value_type raises NotImplementedError."""
    # given
    voucher = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.FIXED,
    )
    VoucherCode.objects.create(code="bad-type", voucher=voucher)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount_value=Decimal(10),
        currency=channel_USD.currency_code,
    )
    # bypass DB validation by setting in-memory
    voucher.discount_value_type = "bad"

    # when & then
    with pytest.raises(NotImplementedError):
        voucher.get_discount(channel_USD)


def test_get_voucher_code_instance_valid(channel_USD):
    """get_voucher_code_instance returns code for active valid voucher."""
    # given
    voucher = Voucher.objects.create(
        type=VoucherType.ENTIRE_ORDER,
        discount_value_type=DiscountValueType.FIXED,
    )
    code = "valid-code-test"
    code_instance = VoucherCode.objects.create(code=code, voucher=voucher)
    VoucherChannelListing.objects.create(
        voucher=voucher,
        channel=channel_USD,
        discount=Money(10, channel_USD.currency_code),
    )

    # when
    result = get_voucher_code_instance(code, channel_USD.slug)

    # then
    assert result.pk == code_instance.pk
    assert result.code == code
