"""
Recommendations aligned with the final 4-cluster K-Means model.

The original Module 8 work produced 7 business segments. This file consolidates
those strategies into the 4 production ML segments without changing the ML
model or its segment labels.

ML segments:
1. Budget Shoppers
2. Emerging / Mid-Tier Customers
3. High-Value Customers
4. Premium & Campaign-Responsive Customers
"""

MODEL_SEGMENT_NAMES = [
    "Budget Shoppers",
    "Emerging / Mid-Tier Customers",
    "High-Value Customers",
    "Premium & Campaign-Responsive Customers",
]

RECOMMENDATIONS = {
    "Budget Shoppers": {
        "profile": "Lower-value, price-sensitive customers. Focus on affordable entry products, simple offers, and gradual movement toward higher purchase frequency.",
        "business_objective": "Grow engagement, purchase frequency, and customer value.",
        "primary_strategy": "Budget Buyer development",
        "secondary_strategy": "Deal-driven conversion",
        "message": "Great products, budget-friendly prices.",
        "channel": "SMS + social media",
        "frequency": "Bi-weekly",
        "product_recommendation": "Sweets",
        "discount_strategy": "15-20%",
        "loyalty_strategy": "Simple cashback/points on every purchase",
        "cross_sell": "Wine (entry-level)",
        "upsell": "Small upgrade add-ons",
        "bundle": "Starter Bundle",
        "campaign": "Starter Bundle Push",
        "campaign_objective": "Grow engagement & spend",
        "budget_priority": "Low",
        "expected_outcome": "Increased purchase frequency",
        "kpi": "Purchase Frequency",
        "secondary_actions": [
            "Use value-focused bundle discounts for customers showing strong deal dependency.",
            "Use multi-pack or buy-more-save-more offers where appropriate.",
            "Gradually move suitable customers toward mid-tier products."
        ]
    },

    "Emerging / Mid-Tier Customers": {
        "profile": "Established customers with moderate value and room to grow. Focus on loyalty, milestones, cross-selling, and average order value.",
        "business_objective": "Increase average order value and deepen engagement.",
        "primary_strategy": "Steady Regular development",
        "secondary_strategy": "Selective deal-based upgrades",
        "message": "You've been a valued regular — here's something rewarding for you.",
        "channel": "Email + app notification",
        "frequency": "Bi-weekly",
        "product_recommendation": "Meat",
        "discount_strategy": "10-15%",
        "loyalty_strategy": "Points-based rewards program",
        "cross_sell": "Fish",
        "upsell": "Family-size Meat & Fish packs",
        "bundle": "Everyday Essentials Bundle",
        "campaign": "Loyalty Milestones",
        "campaign_objective": "Upgrade to premium tier",
        "budget_priority": "Medium",
        "expected_outcome": "Increased average order value",
        "kpi": "Average Order Value",
        "secondary_actions": [
            "Use milestone rewards to encourage repeat purchases.",
            "Cross-sell complementary categories instead of relying on blanket discounts.",
            "Use selective promotions to move suitable customers toward higher-value purchases."
        ]
    },

    "High-Value Customers": {
        "profile": "High-value customers with strong spending and/or purchase frequency. Protect retention while increasing basket size and share-of-wallet.",
        "business_objective": "Maximize retention, lifetime value, and basket value.",
        "primary_strategy": "Premium loyalty and retention",
        "secondary_strategy": "Frequent-shopper basket expansion",
        "message": "Complete your favorites with these picks.",
        "channel": "App notification + email",
        "frequency": "Weekly",
        "product_recommendation": "Meat",
        "discount_strategy": "10-15% selectively",
        "loyalty_strategy": "Frequency rewards",
        "cross_sell": "Gold Products",
        "upsell": "Subscription/auto-replenish bundles",
        "bundle": "Frequent Shopper Loyalty Box",
        "campaign": "Basket Booster",
        "campaign_objective": "Increase basket size",
        "budget_priority": "Medium",
        "expected_outcome": "Higher revenue per visit",
        "kpi": "Average Basket Size",
        "secondary_actions": [
            "Offer premium or limited-edition products to suitable high-value customers.",
            "Use VIP-style recognition and retention benefits rather than heavy discounting.",
            "Cross-sell Gold Products and complementary premium categories.",
            "Use curated premium bundles to increase share-of-wallet."
        ]
    },

    "Premium & Campaign-Responsive Customers": {
        "profile": "Highest-priority customers combining strong customer value with strong campaign responsiveness. Use exclusivity, early access, premium offers, and referrals.",
        "business_objective": "Maximize retention, lifetime value, engagement, and referrals.",
        "primary_strategy": "VIP premium loyalty",
        "secondary_strategy": "Campaign-led early access",
        "message": "Be the first to try what's new.",
        "channel": "Email + app push",
        "frequency": "Weekly",
        "product_recommendation": "Meat",
        "discount_strategy": "10% for campaign offers; 5-10% for premium loyalty where appropriate",
        "loyalty_strategy": "VIP / early-access loyalty perks",
        "cross_sell": "Gold Products; Fish and Sweets for campaign-led offers",
        "upsell": "New arrivals, early-access items, and premium/limited-edition bundles",
        "bundle": "Premium Tasting Bundle / New Launch Sampler",
        "campaign": "Early Access Program + VIP Rewards",
        "campaign_objective": "Deepen engagement and referrals while protecting loyalty",
        "budget_priority": "High",
        "expected_outcome": "Higher repeat purchase and referral volume",
        "kpi": "Repeat Purchase Rate / Referral Conversion Rate",
        "secondary_actions": [
            "Provide exclusive access to the Premium Collection.",
            "Use early access for new products and limited launches.",
            "Favor premium gifts or VIP events over aggressive price reductions.",
            "Use high-value referral rewards to turn engaged customers into advocates."
        ]
    }
}

# The original Module 8 business segments are retained for traceability.
SOURCE_TO_ML_SEGMENT = {
    "Premium Loyal Customers": "Premium & Campaign-Responsive Customers",
    "Steady Regulars": "Emerging / Mid-Tier Customers",
    "Deal-Driven Shoppers": "Budget Shoppers",
    "Frequent Shoppers": "High-Value Customers",
    "Budget Buyers": "Budget Shoppers",
    "Campaign-Engaged Customers": "Premium & Campaign-Responsive Customers",
    # This one is handled as a retention trigger because the 4-cluster ML
    # model does not produce a dedicated At-Risk cluster.
    "At-Risk / Inactive Customers": "Retention Trigger",
}

RETENTION_TRIGGER = {
    "label": "At-Risk / Inactive retention trigger",
    "message": "We miss you — here's 20% off to welcome you back.",
    "channel": "Email + retargeting ads",
    "frequency": "Monthly (3-touch sequence)",
    "discount": "20-30% (win-back)",
    "loyalty": "Reactivation bonus points on return",
    "cross_sell": "Gold Products",
    "upsell": "Personalized past-favorites bundle",
    "bundle": "Welcome-Back Bundle",
    "kpi": "Reactivation Rate",
}

def get_business_recommendation(name):
    """Return recommendations for one of the four final ML segments."""
    return RECOMMENDATIONS[name]

def list_business_segments():
    """Return the four ML segment names used by the production model."""
    return list(RECOMMENDATIONS.keys())

def get_recommendation_for_cluster(name, recency=None):
    """
    Return the ML-segment recommendation.

    A high recency value does NOT change the model's segment label. It only
    adds the Module 8 win-back strategy as a retention action.
    """
    recommendation = dict(RECOMMENDATIONS[name])
    recommendation["retention_alert"] = recency is not None and float(recency) >= 60
    if recommendation["retention_alert"]:
        recommendation["retention_strategy"] = RETENTION_TRIGGER
    return recommendation
