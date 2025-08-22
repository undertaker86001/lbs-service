"""
隐私保护相关API接口
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session

from app.core.database import get_db, get_redis
from app.core.security import get_current_user_id
from app.services.privacy_service import privacy_service
from app.models.base import ApiResponse

router = APIRouter(prefix="/api/v1/privacy", tags=["隐私保护"])


@router.post("/anonymize", response_model=ApiResponse)
async def anonymize_location(
    latitude: float = Body(..., description="纬度", ge=-90, le=90),
    longitude: float = Body(..., description="经度", ge=-180, le=180),
    user_density: Optional[int] = Body(None, description="用户密度（每平方公里用户数）"),
    area_type: str = Body("urban", description="区域类型"),
    current_user_id: int = Depends(get_current_user_id)
):
    """
    使用G-Casper算法匿名化位置数据
    
    - **latitude**: 原始纬度 (-90 到 90)
    - **longitude**: 原始经度 (-180 到 180)
    - **user_density**: 用户密度（可选）
    - **area_type**: 区域类型 ("urban", "suburban", "rural")
    """
    try:
        # 应用G-Casper算法
        result = privacy_service.apply_g_casper_algorithm(
            latitude=latitude,
            longitude=longitude,
            user_density=user_density,
            area_type=area_type
        )
        
        return ApiResponse(
            success=True,
            message="位置匿名化成功",
            data=result
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"位置匿名化失败: {str(e)}")


@router.post("/batch-anonymize", response_model=ApiResponse)
async def batch_anonymize_locations(
    locations: List[dict] = Body(..., description="位置数据列表"),
    batch_size: int = Body(100, description="批处理大小", ge=1, le=1000),
    current_user_id: int = Depends(get_current_user_id)
):
    """
    批量匿名化位置数据
    
    - **locations**: 位置数据列表，每个位置包含 latitude, longitude, user_density, area_type
    - **batch_size**: 批处理大小，范围1-1000
    """
    try:
        # 验证输入数据
        if not locations:
            raise HTTPException(status_code=400, detail="位置数据列表不能为空")
        
        if len(locations) > 10000:
            raise HTTPException(status_code=400, detail="单次处理的位置数据不能超过10000条")
        
        # 批量匿名化
        results = privacy_service.batch_anonymize_locations(locations, batch_size)
        
        return ApiResponse(
            success=True,
            message=f"批量匿名化完成，共处理 {len(results)} 条记录",
            data={
                "total_processed": len(results),
                "batch_size": batch_size,
                "results": results
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量匿名化失败: {str(e)}")


@router.post("/privacy-report", response_model=ApiResponse)
async def generate_privacy_report(
    original_data: dict = Body(..., description="原始数据"),
    anonymous_data: dict = Body(..., description="匿名化数据"),
    current_user_id: int = Depends(get_current_user_id)
):
    """
    生成隐私保护报告
    
    - **original_data**: 原始位置数据
    - **anonymous_data**: 匿名化后的数据
    """
    try:
        # 生成隐私保护报告
        report = privacy_service.generate_privacy_report(original_data, anonymous_data)
        
        return ApiResponse(
            success=True,
            message="隐私保护报告生成成功",
            data=report
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成隐私保护报告失败: {str(e)}")


@router.post("/validate-settings", response_model=ApiResponse)
async def validate_privacy_settings(
    settings: dict = Body(..., description="隐私设置"),
    current_user_id: int = Depends(get_current_user_id)
):
    """
    验证隐私设置的有效性
    
    - **settings**: 隐私设置，包含 k_anonymity, l_max, l_min, area_types 等
    """
    try:
        # 验证隐私设置
        validation_result = privacy_service.validate_privacy_settings(settings)
        
        return ApiResponse(
            success=True,
            message="隐私设置验证完成",
            data=validation_result
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"验证隐私设置失败: {str(e)}")


@router.get("/statistics", response_model=ApiResponse)
async def get_privacy_statistics(
    current_user_id: int = Depends(get_current_user_id)
):
    """
    获取隐私保护统计信息
    """
    try:
        # 获取统计信息
        stats = privacy_service.get_privacy_statistics()
        
        return ApiResponse(
            success=True,
            message="获取隐私保护统计信息成功",
            data=stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@router.get("/geohash-privacy/{geohash}", response_model=ApiResponse)
async def analyze_geohash_privacy(
    geohash: str,
    current_user_id: int = Depends(get_current_user_id)
):
    """
    分析Geohash编码的隐私保护效果
    
    - **geohash**: Geohash编码
    """
    try:
        # 验证Geohash格式
        if not geohash or len(geohash) > 12:
            raise HTTPException(status_code=400, detail="无效的Geohash编码")
        
        # 获取Geohash信息
        from app.services.geohash_service import geohash_service
        area_info = geohash_service.get_geohash_area(geohash)
        
        # 分析隐私保护效果
        precision = len(geohash)
        privacy_analysis = {
            "geohash": geohash,
            "precision": precision,
            "area_info": area_info,
            "privacy_analysis": {
                "precision_level": _get_precision_level(precision),
                "estimated_users": privacy_service._estimate_users_in_geohash(geohash),
                "k_anonymity_satisfied": privacy_service._check_k_anonymity(geohash),
                "privacy_recommendations": _get_privacy_recommendations(precision)
            }
        }
        
        return ApiResponse(
            success=True,
            message="Geohash隐私分析完成",
            data=privacy_analysis
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析Geohash隐私失败: {str(e)}")


@router.post("/test-anonymization", response_model=ApiResponse)
async def test_anonymization_effect(
    test_locations: List[dict] = Body(..., description="测试位置列表"),
    current_user_id: int = Depends(get_current_user_id)
):
    """
    测试匿名化效果
    
    - **test_locations**: 测试位置列表，每个位置包含 latitude, longitude, area_type
    """
    try:
        # 验证输入数据
        if not test_locations or len(test_locations) > 100:
            raise HTTPException(status_code=400, detail="测试位置数量应在1-100之间")
        
        test_results = []
        
        for location in test_locations:
            try:
                # 应用匿名化
                result = privacy_service.apply_g_casper_algorithm(
                    latitude=location["latitude"],
                    longitude=location["longitude"],
                    area_type=location.get("area_type", "urban")
                )
                
                # 计算精度损失
                precision_loss = len(result["original_geohash"]) - len(result["anonymous_geohash"])
                
                test_results.append({
                    "original_location": location,
                    "anonymous_result": result,
                    "precision_loss": precision_loss,
                    "success": True
                })
                
            except Exception as e:
                test_results.append({
                    "original_location": location,
                    "error": str(e),
                    "success": False
                })
        
        # 计算统计信息
        successful_tests = [r for r in test_results if r["success"]]
        if successful_tests:
            avg_precision_loss = sum(r["precision_loss"] for r in successful_tests) / len(successful_tests)
            success_rate = len(successful_tests) / len(test_results)
        else:
            avg_precision_loss = 0
            success_rate = 0
        
        summary = {
            "total_tests": len(test_results),
            "successful_tests": len(successful_tests),
            "success_rate": round(success_rate, 3),
            "average_precision_loss": round(avg_precision_loss, 2),
            "test_results": test_results
        }
        
        return ApiResponse(
            success=True,
            message="匿名化效果测试完成",
            data=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"测试匿名化效果失败: {str(e)}")


@router.get("/compliance-check", response_model=ApiResponse)
async def check_privacy_compliance(
    current_user_id: int = Depends(get_current_user_id)
):
    """
    检查隐私保护合规性
    """
    try:
        # 获取当前隐私设置
        current_settings = {
            "k_anonymity": privacy_service.k_anonymity,
            "l_max": privacy_service.l_max,
            "l_min": privacy_service.l_min
        }
        
        # 验证设置
        validation_result = privacy_service.validate_privacy_settings(current_settings)
        
        # 检查合规性
        compliance_check = {
            "current_settings": current_settings,
            "validation_result": validation_result,
            "compliance_status": {
                "gdpr": "compliant" if current_settings["k_anonymity"] >= 20 else "non_compliant",
                "ccpa": "compliant" if current_settings["k_anonymity"] >= 10 else "non_compliant",
                "local_regulations": "compliant" if validation_result["is_valid"] else "non_compliant"
            },
            "recommendations": validation_result.get("recommendations", [])
        }
        
        return ApiResponse(
            success=True,
            message="隐私保护合规性检查完成",
            data=compliance_check
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"合规性检查失败: {str(e)}")


def _get_precision_level(precision: int) -> str:
    """获取精度级别描述"""
    if precision >= 10:
        return "超高精度（厘米级）"
    elif precision >= 8:
        return "高精度（米级）"
    elif precision >= 6:
        return "中等精度（百米级）"
    elif precision >= 4:
        return "低精度（公里级）"
    else:
        return "极低精度（十公里级）"


def _get_privacy_recommendations(precision: int) -> List[str]:
    """获取隐私保护建议"""
    recommendations = []
    
    if precision >= 10:
        recommendations.append("当前精度过高，建议降低精度以提高隐私保护")
        recommendations.append("考虑使用8位或更低精度")
    elif precision >= 8:
        recommendations.append("当前精度适中，隐私保护效果良好")
        recommendations.append("可以根据用户密度进一步调整")
    elif precision >= 6:
        recommendations.append("当前精度较低，隐私保护效果很好")
        recommendations.append("可以适当提高精度以改善用户体验")
    else:
        recommendations.append("当前精度很低，隐私保护效果极好")
        recommendations.append("建议提高精度以改善服务质量")
    
    return recommendations
