/*
  Warnings:

  - You are about to drop the column `modelId` on the `Evaluation` table. All the data in the column will be lost.
  - You are about to drop the `Model` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropForeignKey
ALTER TABLE "Evaluation" DROP CONSTRAINT "Evaluation_modelId_fkey";

-- DropForeignKey
ALTER TABLE "Model" DROP CONSTRAINT "Model_userId_fkey";

-- AlterTable
ALTER TABLE "Evaluation" DROP COLUMN "modelId";

-- DropTable
DROP TABLE "Model";
